import os
import re
import json
import time
import argparse
from copy import deepcopy
from util.matching import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", type=str, required=True)
    parser.add_argument("-n", "--new_dir", type=str)
    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    NEW_DIR = args.new_dir
    if NEW_DIR != None:
        print('is Re-Extract method')
        NEW_JSON_DIR = os.path.join(args.new_dir, 'JSON')
    JSON_DIR = os.path.join(ROOT_DIR, 'JSON')

    is_completed = False
    sets_data = []
    start = time.perf_counter()

    if os.path.exists(JSON_DIR):
        try:
            if NEW_DIR is None:
                json_files = [filename for filename in os.listdir(
                    JSON_DIR) if filename.startswith("0")]
                for json_file in sorted(json_files, key=lambda x: int(x[:4])):
                    if re.search('[0-9]{4}.json', json_file):
                        with open(os.path.join(JSON_DIR, json_file), 'r', encoding='utf-8-sig') as fs:
                            set_data = json.loads(fs.read())
                            sets_data.append(set_data)
            else:
                old_sets_data = []
                old_json_files = [filename for filename in os.listdir(
                    JSON_DIR) if filename.startswith("Set-")]
                for json_file in sorted(old_json_files, key=lambda x: int(x[4:-5])):
                    if re.search('Set-[0-9]*.json', json_file):
                        with open(os.path.join(JSON_DIR, json_file), 'r', encoding='utf-8-sig') as fs:
                            set_data = json.loads(fs.read())
                            old_sets_data.append(set_data)

                sets_data.append(old_sets_data)

                new_sets_data = []
                new_json_files = [filename for filename in os.listdir(
                    NEW_JSON_DIR) if filename.startswith("Set-")]
                for new_json_file in sorted(new_json_files, key=lambda x: int(x[4:-5])):
                    if re.search('Set-[0-9]*.json', new_json_file):
                        with open(os.path.join(NEW_JSON_DIR, new_json_file), 'r', encoding='utf-8-sig') as fs:
                            set_data = json.loads(fs.read())
                            new_sets_data.append(set_data)

                sets_data.append(new_sets_data)

            master = sets_data[0]

            # start matching data
            if len(sets_data) > 1:
                for index, slave in enumerate(sets_data[1:]):
                    # set matching
                    set_result, master_index, slave_index = sets_matching(
                        master, slave)
                    # col matching
                    master = col_matching_forDB(
                        set_result, master, slave, master_index, slave_index, model_select=2, re_extraction=(NEW_DIR != None))

                    print(
                        f"Finish matching {index + 1}/{len(sets_data[1:])} data!")
            else:
                print("Only one extractor, unable to combine!!!")

            sets_data = master

            # remove sets_data less than 3
            for index, data in enumerate(deepcopy(sets_data)):
                if len(data) < 3:
                    sets_data.remove(data)

            for set_index, set_data in enumerate(sets_data):
                with open(os.path.join(JSON_DIR, f"Set-{str(set_index)}.json"), 'w+', encoding='utf-8-sig') as fs:
                    json.dump(set_data, fs)

            print(f"DEvOSM execution time: {(time.perf_counter() - start):.2f} s")
        except Exception as e:
            print(e)
