# 00 · Step-by-step getting started (Windows)

Everything happens in the **VS Code terminal** (PowerShell). About 10 minutes.

## 1. Set up the project

Download `quartier-flex-en.zip`, then:

```powershell
Expand-Archive -Path "$HOME\Downloads\quartier-flex-en.zip" -DestinationPath C:\dev -Force
cd C:\dev\quartier-flex-en
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[all]"
code .
```

In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`, then open a
**new terminal** (it must show `(.venv)`).

## 2. Check

```powershell
quartier check
pytest -q
```

`pytest` must show `35 passed`.

## 3. Test the modules ONE BY ONE

```powershell
quartier demo weather
quartier demo solar
quartier demo batteries
quartier demo usage
quartier demo appliances
quartier demo grid
quartier demo demand_response
quartier demo control
quartier demo aggregator
quartier demo measure
```

Charts: `start results\demo_appliances.png` (**close the image viewer before running again**).

## 4. The demand-response assessment

```powershell
quartier flex                           # all strategies, fake LLM agent (fast)
quartier flex --llm ollama              # with the real LLM (a few minutes, close other apps)
```

Options: `--start 2024-01-08`, `--days 5`, `--homes 100`, `--packs 6`, `--kwc 200`.

## 5. The interface

```powershell
quartier interface
```

`Ctrl+C` in the terminal to stop.

## 6. Publish on GitHub (new repository)

1. On https://github.com/new: name `quartier-flex-en`, **Public**, **without** README, .gitignore or license → **Create repository**.
2. In the terminal:

```powershell
cd C:\dev\quartier-flex-en
git init
git add .
git commit -m "Quartier Flex: AI demand response for a district (connected appliances, second-life batteries, solar) - Aclimakathon 2026"
git branch -M main
git remote add origin https://github.com/corinne3/quartier-flex-en.git
git push -u origin main
```
