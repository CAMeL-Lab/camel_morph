# Camel Morph SIGMORPHON 2022

This section guides you through the process of inspecting, making use of, and replicating the results obtained for the SIGMORPHON 2022 Camel Morph paper[^1]. Firstly, all the data can be obtained or viewed as described in the [Data](https://github.com/CAMeL-Lab/camel_morph#data) section in the main [README](../README.md) of the repository. However, all the data and code (including relevant Camel Tools modules) required to replicate the paper results are already contained in the standalone `./sigmorphon2022_release` directory. Furthermore, the generated DBs can only be read by the Camel Tools modules included in the latter directory, and not using the official Camel Tools release. To replicate the paper results, follow the below instructions. For a fuller picture of all configurations, see the [Instructions](https://github.com/CAMeL-Lab/camel_morph#instructions) section in the main [README](../README.md) of the repository.

## Installation

1. Clone (download) this repository and unzip in a directory of your choice.
2. Make sure that you are running a version of Python higher than **Python 3.3** and lower than **Python 3.10**.
3. Run the following command to install all needed libraries: `pip install -r requirements.txt`.
4. Run all commands/scripts from the `sigmorphon2022_release` directory.

## Modern Standard Arabic (MSA) Results

To generate the MSA verbs database, the results of which were described in the paper[^1], run the following two commands from the main repository directory to output the resulting DB (`msa_cam_ready_sigmorphon2022_v1.0.db`) into the `sigmorphon2022_release/databases/camel-morph-msa` directory:

    >> cd sigmorphon2022_release
    >> python db_maker.py -config_file config.json -config_name msa_cam_ready_sigmorphon2022 

## Egyptian Arabic (EGY) Results

To generate the EGY verbs database, the results of which were described in the paper[^1], run the following two commands from the main repository directory to output the resulting DB (`egy_cam_ready_sigmorphon2022_v1.0.db`) into the `sigmorphon2022_release/databases/camel-morph-egy` directory:

    >> cd sigmorphon2022_release
    >> python db_maker.py -config_file config.json -config_name egy_cam_ready_sigmorphon2022

## Dummy Example

The example described in Figure 2 of the paper [^1] was recreated for initiation purposes under the configuration name `msa_example_sigmorphon2022`. The DB for it can be generated in a similar fashion as for the DBs above.

## Analysis and Generation

In order to use the generated DB for analysis or generation, follow the same instructions provided in the examples at the following links. Note that the Camel Tools modules included in the `sigmorphon2022_release` directory are required to be used instead of the official release. As long as all code is ran from inside the latter directory, then all behavior should be similar to actually using the official library:

- [Analysis](https://camel-tools.readthedocs.io/en/latest/api/morphology/analyzer.html)
  - [Disambiguation](https://camel-tools.readthedocs.io/en/latest/api/disambig/mle.html) (in-context analysis)
- [Generation](https://camel-tools.readthedocs.io/en/latest/api/morphology/generator.html)

[^1]: Habash, Nizar et al. "Morphotactic Modeling in an Open-source Multi-dialectal Arabic Morphological Analyzer and Generator", SIGMORPHON. 2022
[^2]: Note that for the release directory, only the morphological components from Camel Tools were sourced from the actual library and were added to be imported locally.
