from pydantic import BaseModel, Field, confloat, PositiveFloat, NonNegativeFloat
import pyomo.environ as pyomo
import pyomo.opt as opt

# ## Entities

# There is a set of ingredients, each with a name, alcohol per volume (abv), and production cost (per gallon).


class Ingredient(BaseModel):
    name: str
    abv: confloat(ge=0, le=1)
    cost: NonNegativeFloat


# There is a desired product with a target alcohol per volume (abv) and volume.


class DesiredProduct(BaseModel):
    volume: PositiveFloat
    abv: confloat(ge=0, le=1)


# The problem is specified by the desired product and a list of possible ingredients.


class Data(BaseModel):
    desired_product: DesiredProduct
    options: list[Ingredient]


# ## Goals and Requirements

# The goal of the problem is to decide how much volume of each ingredient to use


def add_produced_volume_variable(model: pyomo.ConcreteModel, data: Data) -> None:
    model.produced_volume = pyomo.Var(
        [o.name for o in data.options], domain=pyomo.NonNegativeReals
    )


# such that the total cost of producing the desired product is minimized,


def add_objective(model: pyomo.ConcreteModel, data: Data) -> None:
    model.cost = pyomo.Objective(
        sense=pyomo.minimize,
        expr=sum(
            model.produced_volume[option.name] * option.cost for option in data.options
        ),
    )


# while making sure the resulting brew has the desired volume


def add_correct_volume_constraint(model: pyomo.ConcreteModel, data: Data) -> None:
    model.vol = pyomo.Constraint(
        expr=data.desired_product.volume
        == sum(model.produced_volume[option.name] for option in data.options)
    )


# and the desired alcohol per volume.


def add_correct_abv_constraint(model: pyomo.ConcreteModel, data: Data) -> None:
    model.abv = pyomo.Constraint(
        expr=0
        == sum(
            model.produced_volume[option.name] * (option.abv - data.desired_product.abv)
            for option in data.options
        )
    )


# ## Output

# A solution of this problem should specify the total cost of the brew and the volume of each ingredient used.


class SolverOutput(BaseModel):
    termination_condition: opt.TerminationCondition  # = Field(None)
    solver_status: opt.SolverStatus  # = Field(None)
    optimal_cost: float | None = Field(
        default=None, description="Total cost of the brew"
    )
    volumes_of_ingredients_to_produce: dict[str, float] | None = Field(
        default=None, description="Volume of each ingredient used"
    )


def create_instance(data: Data) -> pyomo.ConcreteModel:
    model = pyomo.ConcreteModel()
    add_produced_volume_variable(model=model, data=data)
    add_objective(model=model, data=data)
    add_correct_volume_constraint(model=model, data=data)
    add_correct_abv_constraint(model=model, data=data)
    return model


def solve_func(data: Data, solver_name: str = "glpk") -> SolverOutput:
    instance = create_instance(data=data)
    solver = opt.SolverFactory(solver_name)
    results: opt.SolverResults = solver.solve(instance)
    print(results.solver.termination_condition)
    if results.solver.termination_condition not in [
        opt.TerminationCondition.optimal,
        opt.TerminationCondition.feasible,
    ]:
        produced_volume = None
        objective_value = None

    else:
        produced_volume = {
            o: pyomo.value(instance.produced_volume[o])
            for o in instance.produced_volume
        }
        objective_value = pyomo.value(instance.cost)

    return SolverOutput(
        termination_condition=results.solver.termination_condition,
        solver_status=results.solver.status,
        optimal_cost=objective_value,
        volumes_of_ingredients_to_produce=produced_volume,
    )
