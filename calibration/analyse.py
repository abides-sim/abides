import optuna


def visualise(study):
    """ visualise the study results using optuna visualization

    :param study:
    :return: None
    """
    slice = optuna.visualization.plot_slice(study)
    parallel_coordinates = optuna.visualization.plot_parallel_coordinate(study)
    optimization_history = optuna.visualization.plot_optimization_history(study)
    contour = optuna.visualization.plot_contour(study, params=['n_value', 'n_noise'])
    slice.write_image("results/slice.png")
    parallel_coordinates.write_image("results/parallel_coordinates")
    optimization_history.write_image("results/optimization_history.png")
    contour.write_image("results/contour.png")


if __name__ == "__main__":

    study_name = 'abides_study'
    storage = f'sqlite:///{study_name}.db'

    study = optuna.study.load_study(study_name=study_name,
                                    storage=storage)

    visualise(study)