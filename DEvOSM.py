import os
import re
import json
import argparse
from util.matching import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", type=str, required=True)
    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    JSON_DIR = os.path.join(ROOT_DIR, 'JSON')

    is_completed = False
    sets_data = []

    if os.path.exists(JSON_DIR):
        try:
            for json_file in os.listdir(JSON_DIR):
                if re.search('[0-9]{4}.json', json_file):
                    with open(os.path.join(JSON_DIR, json_file), 'r', encoding='utf-8-sig') as fs:
                        set_data = json.loads(fs.read())
                        sets_data.append(set_data)

            master = sets_data[0]

            # start matching data
            if len(sets_data) > 1:
                for slave in sets_data[1:]:
                    # set matching
                    set_result, master_index, slave_index = sets_matching(master, slave)
                    # col matching
                    master = col_matching_forDB(set_result, master, slave, master_index, slave_index, model_select=2)
            else:
                raise Exception("Unable to combine!!!")

            sets_data = master

            # remove sets_data less than 3
            for index, data in enumerate(sets_data.copy()):
                if len(data) < 3:
                    sets_data.remove(data)

            for set_index, set_data in enumerate(sets_data):
                with open(os.path.join(JSON_DIR, f"Set-{str(set_index)}.json"), 'w+', encoding='utf-8-sig') as fs:
                    json.dump(set_data, fs)          
        except Exception as e:
            print(e)
