import os
import sys
import json
import asyncio
from typing import Any, Dict, List

import dotenv
from github import Github, Auth, GithubException

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, AgentOutput, ToolCall, ToolCallResult
from llama_index.core.workflow import Context


# ----------------------------
# ENV helpers
# ----------------------------
def getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def normalize_pr_number(v: str | None) -> int:
    if not v:
        raise ValueError("PR_NUMBER is missing")
    return int(v.strip())


def normalize_repository(v: str | None) -> str:
    if not v:
        raise ValueError("REPOSITORY is missing (expected 'owner/repo')")
    return v.strip()


# ----------------------------
# GitHub helpers
# ----------------------------
def build_github_client(token: str | None) -> Github:
    if token:
        return Github(auth=Auth.Token(token))
    return Github()


def get_repo(gh: Github, full_repo_name: str):
    return gh.get_repo(full_repo_name)


def get_pr_details(gh: Github, full_repo_name: str, pr_number: int) -> Dict[str, Any]:
    repo = get_repo(gh, full_repo_name)
    pr = repo.get_pull(pr_number)

    # head_sha: take latest commit sha from PR
    head_sha = pr.head.sha if pr.head and pr.head.sha else ""

    return {
        "author": pr.user.login if pr.user else "",
        "title": pr.title or "",
        "body": pr.body or "",  # must be present even if empty
        "diff_url": pr.diff_url or "",
        "state": pr.state or "",
        "head_sha": head_sha,
    }


def get_pr_files(gh: Github, full_repo_name: str, pr_number: int) -> List[Dict[str, Any]]:
    repo = get_repo(gh, full_repo_name)
    pr = repo.get_pull(pr_number)

    changed = []
    for f in pr.get_files():
        changed.append(
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": getattr(f, "patch", None),
            }
        )
    return changed


def post_review_or_comment(gh: Github, full_repo_name: str, pr_number: int, comment: str) -> Dict[str, Any]:
    repo = get_repo(gh, full_repo_name)
    pr = repo.get_pull(pr_number)

    title = pr.title or ""
    body = pr.body or ""

    try:
        pr.create_review(body=comment)
        return {
            "status": "posted_review",
            "pr_number": pr_number,
            "title": title,
            "body": body,
        }
    except GithubException as e:
        # 422: pending review exists
        if getattr(e, "status", None) == 422:
            pr.create_issue_comment(comment)
            return {
                "status": "posted_issue_comment",
                "pr_number": pr_number,
                "title": title,
                "body": body,
                "note": "Posted as issue comment because a pending review already exists.",
            }

        return {
            "status": "error",
            "http_status": getattr(e, "status", None),
            "data": getattr(e, "data", None),
            "pr_number": pr_number,
            "title": title,
            "body": body,
        }


# ----------------------------
# In-memory state (simple dict)
# ----------------------------
_STATE: Dict[str, Any] = {
    "pr_number": None,
    "repository": "",
    "gathered_contexts": "",
    "review_comment": "",
    "final_review_comment": "",
}


async def get_state(_: Context) -> Dict[str, Any]:
    return _STATE


async def set_pr_number_in_state(_: Context, pr_number: int) -> str:
    _STATE["pr_number"] = pr_number
    return "Saved pr_number to state."


async def set_repository_in_state(_: Context, repository: str) -> str:
    _STATE["repository"] = repository
    return "Saved repository to state."


async def add_context_to_state(_: Context, gathered_contexts: str) -> str:
    _STATE["gathered_contexts"] = gathered_contexts
    return "Saved gathered_contexts to state."


async def add_review_comment_to_state(_: Context, review_comment: str) -> str:
    _STATE["review_comment"] = review_comment
    return "Saved review_comment to state."


async def add_final_review_to_state(_: Context, final_review_comment: str) -> str:
    _STATE["final_review_comment"] = final_review_comment
    return "Saved final_review_comment to state."


# ----------------------------
# Tools that use GH client
# ----------------------------
def tool_get_pr_details() -> Dict[str, Any]:
    gh = build_github_client(getenv("GITHUB_TOKEN"))
    repo = normalize_repository(getenv("REPOSITORY"))
    pr_number = normalize_pr_number(getenv("PR_NUMBER"))
    return get_pr_details(gh, repo, pr_number)


def tool_get_pr_files() -> List[Dict[str, Any]]:
    gh = build_github_client(getenv("GITHUB_TOKEN"))
    repo = normalize_repository(getenv("REPOSITORY"))
    pr_number = normalize_pr_number(getenv("PR_NUMBER"))
    return get_pr_files(gh, repo, pr_number)


def tool_post_review(comment: str) -> Dict[str, Any]:
    gh = build_github_client(getenv("GITHUB_TOKEN"))
    repo = normalize_repository(getenv("REPOSITORY"))
    pr_number = normalize_pr_number(getenv("PR_NUMBER"))
    return post_review_or_comment(gh, repo, pr_number, comment)


# ----------------------------
# Agents
# ----------------------------
context_system_prompt = """
You are the context gathering agent.

WORKFLOW RULES (VERY IMPORTANT):
- Call get_state first.
- Call get_pr_details and get_pr_files using tools.
- Build ONE gathered_contexts JSON string containing:
  author, title, body, diff_url, state, head_sha, changed_files.
- Call add_context_to_state exactly once with gathered_contexts.
- Then handoff to CommentorAgent and STOP.
"""

commentor_system_prompt = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.

You must use the context stored in state.gathered_contexts to draft a PR review comment.

REVIEW REQUIREMENTS:
- Write a ~200-300 word review in markdown format.
- Include: what is good, contribution rules compliance/missing items, tests/migrations notes, endpoint documentation notes,
  and quote specific lines that could be improved with suggestions.
- Address the author directly.

WORKFLOW RULES (VERY IMPORTANT):
- First call get_state.
- If state.gathered_contexts is empty, you MUST handoff to ContextAgent and STOP.
- If state.gathered_contexts is available:
  - Draft the review comment now.
  - Call add_review_comment_to_state exactly once with the draft.
  - Then handoff to ReviewAndPostingAgent and STOP.
"""

reviewer_system_prompt = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
- The review must:
  - Be a ~200-300 word review in markdown format.
  - Specify what is good about the PR.
  - Did the author follow ALL contribution rules? What is missing?
  - Notes on test availability; if new models, are there migrations.
  - Notes on whether new endpoints were documented.
  - Suggestions on which lines could be improved; these lines are quoted.

WORKFLOW RULES (VERY IMPORTANT):
- First call get_state and inspect state.review_comment.
- If state.review_comment is empty or does not meet criteria, handoff to CommentorAgent with rewrite instructions and STOP.
- Otherwise call add_final_review_to_state exactly once.
- Then call post_review_to_github exactly once with the final markdown comment.
- Then provide the final response and STOP.
"""


def build_llm() -> OpenAI:
    return OpenAI(
        model=getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=getenv("OPENAI_API_KEY", ""),
        api_base=getenv("OPENAI_BASE_URL"),
    )


def main_workflow() -> AgentWorkflow:
    llm = build_llm()

    # Tools
    get_state_tool = FunctionTool.from_defaults(get_state)
    set_pr_tool = FunctionTool.from_defaults(set_pr_number_in_state)
    set_repo_tool = FunctionTool.from_defaults(set_repository_in_state)
    add_ctx_tool = FunctionTool.from_defaults(add_context_to_state)
    add_review_tool = FunctionTool.from_defaults(add_review_comment_to_state)
    add_final_tool = FunctionTool.from_defaults(add_final_review_to_state)

    pr_details_tool = FunctionTool.from_defaults(tool_get_pr_details)
    pr_files_tool = FunctionTool.from_defaults(tool_get_pr_files)
    post_review_tool = FunctionTool.from_defaults(tool_post_review)

    context_agent = FunctionAgent(
        llm=llm,
        name="ContextAgent",
        description="Gathers PR details and changed files and saves context into state.",
        tools=[get_state_tool, pr_details_tool, pr_files_tool, add_ctx_tool],
        system_prompt=context_system_prompt,
        can_handoff_to=["CommentorAgent"],
    )

    commentor_agent = FunctionAgent(
        llm=llm,
        name="CommentorAgent",
        description="Drafts a PR review comment using gathered contexts.",
        tools=[get_state_tool, add_review_tool],
        system_prompt=commentor_system_prompt,
        can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
    )

    review_agent = FunctionAgent(
        llm=llm,
        name="ReviewAndPostingAgent",
        description="Validates draft review, stores final version, and posts to GitHub.",
        tools=[get_state_tool, add_final_tool, post_review_tool],
        system_prompt=reviewer_system_prompt,
        can_handoff_to=["CommentorAgent"],
    )

    workflow = AgentWorkflow(
        agents=[context_agent, commentor_agent, review_agent],
        root_agent=review_agent.name,
        initial_state=_STATE,
    )
    return workflow


async def run():
    dotenv.load_dotenv()

    # GitHub Actions passes env; also supports CLI args as described in task (optional)
    # run: python agent.py $GITHUB_TOKEN $REPOSITORY $PR_NUMBER $OPENAI_API_KEY $OPENAI_BASE_URL
    # We'll map argv into env to match your local runs too.
    argv = sys.argv[1:]
    if len(argv) >= 1 and not getenv("GITHUB_TOKEN"):
        os.environ["GITHUB_TOKEN"] = argv[0]
    if len(argv) >= 2 and not getenv("REPOSITORY"):
        os.environ["REPOSITORY"] = argv[1]
    if len(argv) >= 3 and not getenv("PR_NUMBER"):
        os.environ["PR_NUMBER"] = argv[2]
    if len(argv) >= 4 and not getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = argv[3]
    if len(argv) >= 5 and not getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = argv[4]

    repository = normalize_repository(getenv("REPOSITORY"))
    pr_number = normalize_pr_number(getenv("PR_NUMBER"))

    # set in state for completeness (not strictly needed, but useful)
    _STATE["repository"] = repository
    _STATE["pr_number"] = pr_number

    query = f"Write a review for PR number {pr_number}."

    workflow = main_workflow()
    ctx = Context(workflow)
    handler = workflow.run(query, ctx=ctx)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response and event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            # print raw so CI logs show it plainly
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

    await handler


if __name__ == "__main__":
    asyncio.run(run())
