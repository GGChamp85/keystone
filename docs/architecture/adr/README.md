# Architecture decision records

Short, dated records of the decisions that shape Keystone, with the context that made them and the consequences they carry. A decision is `accepted` when the code follows it, `proposed` until then.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-ctags-not-tree-sitter.md) | Repository map built with universal-ctags, not tree-sitter | accepted |
| [0002](0002-ray-train-over-kubeflow.md) | Multi-GPU training via Ray Train on KubeRay, not Kubeflow | accepted |
| [0003](0003-llama-cpp-ci-backend.md) | A real CPU-only model backend for CI (llama.cpp, 0.5B GGUF) | accepted |
| [0004](0004-peppered-api-key-hashing.md) | API keys hashed with a peppered HMAC, with rotation | proposed |
| [0005](0005-runpod-serverless-via-rest.md) | RunPod Serverless endpoints created through RunPod's REST API | accepted |
| [0006](0006-vendor-neutral-display-strings.md) | Repo-facing text stays vendor-neutral | accepted |

To add one: copy the shape of an existing record (Context → Decision → Consequences), number it next, and link it here.
