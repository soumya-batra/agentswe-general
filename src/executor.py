from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from agent import Agent, classify, extract_payload


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class Executor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, Agent] = {} # context_id to agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        # Peek-classify the message to detect CAR-bench, which uses a
        # Message-only response model (no Task lifecycle). Sticky:
        # if this context already has an agent with a handler set,
        # USE that handler — never reclassify mid-conversation. That
        # protects every other benchmark from a future turn whose
        # text or payload coincidentally matches another handler's
        # signature.
        ctx_id_from_msg = getattr(msg, "context_id", None) or ""
        existing_agent = self.agents.get(ctx_id_from_msg) if ctx_id_from_msg else None
        if existing_agent and existing_agent.handler:
            handler_hint = existing_agent.handler
        else:
            payload_peek, raw_text_peek = extract_payload(msg)
            handler_hint = classify(payload_peek, raw_text_peek)

        if handler_hint == "car_bench":
            context_id = ctx_id_from_msg or uuid4().hex
            agent = existing_agent
            if not agent:
                agent = Agent()
                agent.handler = "car_bench"
                self.agents[context_id] = agent
            try:
                await agent.run_car_bench(msg, event_queue, context_id)
            except Exception as e:
                print(f"car_bench failed: {e}")
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"Agent error: {e}", context_id=context_id
                    )
                )
            return

        # Standard Task-style flow for every other handler.
        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} already processed (state: {task.status.state})"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = Agent()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context_id, task_id=task.id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
