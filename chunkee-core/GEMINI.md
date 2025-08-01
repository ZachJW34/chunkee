# Gemini Assistant Guidelines

## Rules

1. If I encounter code, logic, or instructions that appear incorrect, ambiguous, or confusing, I must stop and ask the user for clarification. I will explain what I find confusing and wait for a response before proceeding with any actions or modifications.
2. When using file modification tools like `replace` or `write_file`, the `new_string` or `content` arguments must contain *only* the raw, valid code for the target file. I must never include formatting artifacts, diff markers (like `'''`), or any other syntax that is not part of the code itself. All code provided to these tools must be syntactically correct and ready for direct insertion or writing.
3. When I am asked to perform a task, I will base my work on the state of the code provided in the prompt or by reading the files. I will not assume my previous refactors or changes have been kept by the user.
4. I will not remove or modify any old or debug code unless explicitly instructed to do so by the user.