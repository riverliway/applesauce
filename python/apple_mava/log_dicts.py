from jumanji_env.environments.complex_orchard.constants import TICK_SPEED

def create_complex_dict(state, seed) -> dict:
    """
    Creates a dictionary from the state object.
    """

    bots = [{
        "x": position[0],
        "y": position[1],
        "diameter": diameter,
        "holding": None if holding == -1 else holding,
        "job": 'picker' if job < 0.5 else 'pusher',
        "orientation": orientation
    }
    for position, orientation, holding, job, diameter in zip(
        state.bots.position.tolist(),
        state.bots.orientation.tolist(),
        state.bots.holding.tolist(),
        state.bots.job.tolist(),
        state.bots.diameter.tolist(),
    )]

    trees = [{
        "x": position[0],
        "y": position[1],
        "diameter": diameter
    }
    for position, diameter in zip(
        state.trees.position.tolist(),
        state.trees.diameter.tolist()
    )]

    baskets = [{
        "x": position[0],
        "y": position[1],
        "diameter": diameter,
        "held": False,
        "collected": False
    }
    for position, diameter in zip(
        state.baskets.position.tolist(),
        state.baskets.diameter.tolist()
    )]

    apples = [{
        "x": position[0],
        "y": position[1],
        "diameter": diameter,
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