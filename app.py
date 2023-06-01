from fastapi import FastAPI
from main import Data, SolverOutput, solve_func

app = FastAPI()

@app.get("/")
def run_model(data: Data) -> SolverOutput:
    return solve_func(data)
