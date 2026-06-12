Pull request: $repo#$number — $title

Human review comment (the reference):
[$path:$line] $body

Agent comments (candidates, by index):
$agent_comments

Did any agent comment identify the same underlying issue as the human
comment?

Respond with ONLY this JSON object:
{"verdict": "match|partial|miss", "matched_agent_index": <int or null>, "reason": "<one sentence>"}
