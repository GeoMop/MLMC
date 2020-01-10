import os
import sys
import yaml
import pickle
import json

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))
sys.path.append(os.path.join(src_path, '..', '..', 'test'))
sys.path.append(os.path.join(src_path))

print("sys.path ", sys.path)


def build():
    job_id = sys.argv[2]
    files = read_file_structure(sys.argv[1])

    level_sims = get_level_config(files['levels_config'])

    scheduled_file_path = files['scheduled'].format(job_id)
    level_id_sample_id = get_level_id_sample_id(scheduled_file_path)

    # Run sample simulation
    results_file_path = files['results'].format(job_id)
    run_sample(level_id_sample_id, level_sims, results_file_path)


def read_file_structure(file_path):
    with open(file_path, "r") as reader:
        files = json.load(reader)
    return files


def get_level_config(levels_config_path):
    """
    Deserialize LevelSimulation object
    :return:
    """
    with open(levels_config_path, "r") as reader:
        level_config_files = reader.read().splitlines()

    level_simulations = {}
    for level_config_file_path in level_config_files:
        print(level_config_file_path)
        with open(level_config_file_path, "rb") as f:
            l_sim = pickle.load(f)
            level_simulations[l_sim.level_id] = l_sim

    return level_simulations


def get_level_id_sample_id(scheduled_path_file):
    with open(scheduled_path_file) as file:
        level_id_sample_id = yaml.load(file)

    return level_id_sample_id


def run_sample(level_id_sample_id, level_sims, result_file_path):
    for level_id, sample_id in level_id_sample_id:
        level_sim = level_sims[level_id]

        assert level_sim.level_id == level_id
        result = level_sim.calculate(level_sim.config_dict, level_sim.sample_workspace)

        mes = ""
        res = [sample_id, [result[0].tolist(), result[1].tolist()], mes]

        write_results_to_file(res, result_file_path)


def write_results_to_file(result, result_file_path):
    with open(result_file_path, "a") as f:
        yaml.dump(result, f)

    # with open(result_file_path, "r") as f:
    #     result = yaml.load(f)
    #     print("result ", result)


if __name__ == "__main__":
    build()
