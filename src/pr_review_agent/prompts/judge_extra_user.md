Pull request: $repo#$number — $title

The human reviewers' comments on this PR were:
$human_comments

This agent comment matched none of them:
[$file_path:$line] $body

Classify it:
- "plausible-extra": correct and actionable even though no human raised it
- "false-positive": incorrect, confused, trivial, or not actionable

Respond with ONLY this JSON object:
{"verdict": "plausible-extra|false-positive", "reason": "<one sentence>"}
