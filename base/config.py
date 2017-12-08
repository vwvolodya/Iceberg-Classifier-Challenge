import os
import errno
from tensorboardX import SummaryWriter


class ProjectConfig:
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    logger_directory = os.path.join(base_dir, "logs")
    model_directory = os.path.join(base_dir, "models")
    fold_number = 4
    data_directory = os.path.join(base_dir, "data")
    fold_directory = os.path.join(data_directory, "folds")
    vis_directory = os.path.join(data_directory, "vis")
    orig_data_directory = os.path.join(data_directory, "orig")
    logger_class = SummaryWriter

    @classmethod
    def combine(cls, base_path, *pathes):
        new_path = os.path.join(base_path, *pathes)
        return new_path

    @classmethod
    def _make_dir(cls, path):
        try:
            os.makedirs(path)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise
            print("Directory %s already exists!" % path)

    @classmethod
    def make_project_structure(cls):
        cls._make_dir(cls.data_directory)
        cls._make_dir(cls.fold_directory)
        cls._make_dir(cls.logger_directory)
        for f in range(cls.fold_number):
            name = os.path.join(cls.logger_directory, str(f))
            cls._make_dir(name)
        cls._make_dir(cls.vis_directory)
        cls._make_dir(cls.orig_data_directory)
        cls._make_dir(cls.model_directory)


class ExperimentConfig:
    def __init__(self, dataset_class, model_class):
        pass


if __name__ == "__main__":
    ProjectConfig.make_project_structure()
    print()
