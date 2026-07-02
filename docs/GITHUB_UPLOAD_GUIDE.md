# GitHub Upload Guide

## 1. Create an Empty Repository

Create a new public GitHub repository, for example:

```text
ADI-LoRA
```

Recommended description:

```text
Core code, processed tables, figure data, and analysis scripts for ADI-LoRA.
```

## 2. Initialize and Push

From the parent directory of this release folder:

```bash
cd ADI-LoRA
git init
git add .
git commit -m "Initial public release for ADI-LoRA"
git branch -M main
git remote add origin https://github.com/USER_OR_ORG/ADI-LoRA.git
git push -u origin main
```

## 3. Update the Manuscript URL

Replace the manuscript placeholder:

```text
https://github.com/USER_OR_ORG/ADI-LoRA
```

with:

```text
https://github.com/USER_OR_ORG/ADI-LoRA
```

Suggested sentence:

```text
The processed experimental tables, figure data, cleaned implementation code,
and analysis scripts supporting the findings of this study are publicly
available at https://github.com/USER_OR_ORG/ADI-LoRA.
```

## 4. Final Checks Before Public Release

Run:

```bash
find . -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.safetensors" -o -name "*.tar.gz" -o -name "*.zip" \)
```

This should print nothing.

Also confirm that no private paths, tokens, or unpublished raw data are included.
