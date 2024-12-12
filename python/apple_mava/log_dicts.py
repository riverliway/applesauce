from jumanji_env.environments.complex_orchard.constants import TICK_SPEED

TRUNCATION_DECIMALS = 2

def create_complex_dict(state, seed) -> dict:
    """
    Creates a dictionary from the state object.
    """

    bots = [{
        "x": truncate_float(position[0], TRUNCATION_DECIMALS),
        "y": truncate_float(position[1], TRUNCATION_DECIMALS),
        "diameter": truncate_float(diameter, TRUNCATION_DECIMALS),
        "holding": None if holding == -1 else holding,
        "job": 'picker' if job < 0.5 else 'pusher',
        "orientation": truncate_float(orientation, TRUNCATION_DECIMALS + 2)
    }
    for position, orientation, holding, job, diameter in zip(
        state.bots.position.tolist(),
        state.bots.orientation.tolist(),
        state.bots.holding.tolist(),
        state.bots.job.tolist(),
        state.bots.diameter.tolist(),
    )]

    trees = [{
        "x": truncate_float(position[0], TRUNCATION_DECIMALS),
        "y": truncate_float(position[1], TRUNCATION_DECIMALS),
        "diameter": truncate_float(diameter, TRUNCATION_DECIMALS),
    }
    for position, diameter in zip(
        state.trees.position.tolist(),
        state.trees.diameter.tolist()
    )]

    baskets = [{
        "x": truncate_float(position[0], TRUNCATION_DECIMALS),
        "y": truncate_float(position[1], TRUNCATION_DECIMALS),
        "diameter": truncate_float(diameter, TRUNCATION_DECIMALS),
        "held": False,
        "collected": False
    }
    for position, diameter in zip(
        state.baskets.position.tolist(),
        state.baskets.diameter.tolist()
    )]

    apples = [{
        "x": truncate_float(position[0], TRUNCATION_DECIMALS),
        "y": truncate_float(position[1], TRUNCATION_DECIMALS),
        "diameter": truncate_float(diameter, TRUNCATION_DECIMALS),
        "held": held,
        "collected": collected
    }
    for position, diameter, held, collected in zip(
        state.apples.position.tolist(),
        state.apples.diameter.tolist(),
        state.apples.held.tolist(),
        state.apples.collected.tolist()
    )]

    return {
        "width": state.width.item(),
        "height": state.height.item(),
        "seed": seed,
        "time": state.step_count.item(),
        "bots": bots,
        "trees": trees,
        "baskets": baskets,
        "apples": apples,
        "TICK_SPEED": TICK_SPEED
    }

def truncate_float(number: float, decimal_places: int) -> float:
    """
    Truncates a float to a certain number of decimal places.
    """

    return int(number * 10 ** decimal_places) / 10 ** decimal_places