# ----- VN START -----
TEST_CONFIG_FILE = "options/test_editguard.yml"
TRAIN_BIT_CONFIG_FILE = "options/train_editguard_bit.yml"

import yaml
def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Global configuration variable
TEST_CONFIG = load_config(TEST_CONFIG_FILE)
TRAIN_BIT_CONFIG = load_config(TRAIN_BIT_CONFIG_FILE)
# ----- VN END -----