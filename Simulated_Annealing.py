from _collections_abc import Callable, Iterable


def Simulated_Annealing[T](
    initial_state: T,
    Random_Neighbor_Generator: Callable[[T], T],
    State_Entropy_Function: Callable[[T], float],
    annealing_schedule: Iterable[float]
) -> T:
    '''
    The function Apply_Simulated_Annealing takes in the parameters:
        -initial_state
            The initial state, ideally initialized to a close-to-optimal state.

        -Random_Number_Generator
            A function that generates a random neighbor from a given state.

        -State_Entropy_Function
            A function that calculates the entropy of a state.

        -annealing_schedule
            An iterable which contains the annealing schedule.

    The function returns the minimum entropy state
    '''
    from math import exp
    from copy import deepcopy

    from random import SystemRandom
    rand = SystemRandom(0)

    optimum_state = deepcopy(initial_state)

    for temperature in annealing_schedule:
        candidate_state = Random_Neighbor_Generator(optimum_state)
        candidate_state_entropy = State_Entropy_Function(candidate_state)
        optimum_state_entropy = State_Entropy_Function(optimum_state)

        if (
            candidate_state_entropy < optimum_state_entropy or rand.random() <
            exp(-candidate_state_entropy - optimum_state_entropy / temperature)
        ):
            optimum_state = candidate_state

    return optimum_state
