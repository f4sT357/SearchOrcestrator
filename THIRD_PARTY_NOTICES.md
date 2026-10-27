# Third-Party Notices

This project is licensed under the Apache License 2.0. This file records the
primary third-party components used by the application and the notices that
must accompany a redistributed GUI package.

The versions below were verified from the project's runtime environment on
2026-10-27. Update this file and the included license texts when dependencies
are upgraded or the release environment changes.

| Component | Role | License |
| --- | --- | --- |
| DDGS | Web search client | MIT |
| HTTPX | HTTP client | BSD-3-Clause |
| LangChain | LLM application framework | MIT |
| LangChain OpenAI | OpenAI-compatible LangChain integration | MIT |
| LangGraph | Workflow graph framework | MIT |
| lxml | HTML parsing | BSD-3-Clause |
| Pydantic | Data validation | MIT |
| Sentence Transformers | Cross-encoder loading and inference | Apache-2.0 |
| PyTorch | Tensor execution used by Sentence Transformers | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| PySide6 / Qt for Python | Desktop GUI | LGPL-3.0-only, or a Qt commercial license |
| Qt libraries and plugins bundled by PySide6 | Desktop GUI runtime | LGPL-3.0-only, or a Qt commercial license; may include separately licensed third-party materials |
| BAAI/bge-reranker-v2-m3 | Downloaded reranking model | Apache-2.0 |

## GUI redistribution requirements

When distributing the desktop GUI (for example as a directory, installer, or
frozen executable), include this file and `LICENSES/LGPL-3.0.txt` with the
distribution. Retain the license and copyright notices shipped with the
PySide6/Qt runtime and any Qt plugins that are included in the release.

This project uses the open-source LGPL option for PySide6/Qt. The application
must load the Qt runtime as shared DLLs; do not statically link Qt. Recipients
must be allowed to replace the LGPL-covered Qt libraries for debugging their
modifications, as required by LGPL-3.0. If those conditions cannot be met, use
a Qt commercial license instead.

The GUI source and its application code are not required to be relicensed
under LGPL solely because they use PySide6, provided the LGPL obligations above
are met.

## Optional Firecrawl integration

The Firecrawl self-hosting scripts download Firecrawl separately and do not
vendor it in this repository. Firecrawl is licensed primarily under AGPL-3.0.
Do not bundle, redistribute, or modify Firecrawl or its container images as
part of a product release without separately meeting its license obligations.

## Scope

This is a distribution notice, not legal advice. A final release should be
checked against the exact dependency lockfile and packaged files, including all
transitive dependencies and Qt plugins actually shipped.