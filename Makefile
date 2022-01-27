make_dbs:
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_msa_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_msa_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_msa_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_glf_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_glf_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_glf_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_egy_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_egy_order-v4_red
	python db_maker.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_egy_order-v4_red

repr_verbs:
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_msa_order-v4_red -cmplx_morph "[STEM-PV]" -output_name conjugation/repr_verbs/repr_verbs_pv_msa.csv
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_msa_order-v3 -cmplx_morph "[STEM-IV]" -output_name conjugation/repr_verbs/repr_verbs_iv_msa.csv
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_msa_order-v4_red -cmplx_morph "[STEM-CV]" -output_name conjugation/repr_verbs/repr_verbs_cv_msa.csv
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_glf_order-v4_red -cmplx_morph "[STEM-PV]" -output_name conjugation/repr_verbs/repr_verbs_pv_glf.csv
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_glf_order-v4_red -cmplx_morph "[STEM-IV]" -output_name conjugation/repr_verbs/repr_verbs_iv_glf.csv
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_glf_order-v4_red -cmplx_morph "[STEM-CV]" -output_name conjugation/repr_verbs/repr_verbs_cv_glf.csv
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name pv_egy_order-v4_red -cmplx_morph "[STEM-PV]" -output_name conjugation/repr_verbs/repr_verbs_pv_egy.csv
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name iv_egy_order-v4_red -cmplx_morph "[STEM-IV]" -output_name conjugation/repr_verbs/repr_verbs_iv_egy.csv
	python create_repr_verb_list.py -specs_sheets specs_sheets/CamelMorphDB-v1.0.xlsx -config_file config.json -config_name cv_egy_order-v4_red -cmplx_morph "[STEM-CV]" -output_name conjugation/repr_verbs/repr_verbs_cv_egy.csv

conj_tables_msa:
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_pv_msa.csv -db db_iterations/XYZ_pv_v2.0_red.db -asp p -dialect msa -output_name conjugation/tables/conj_pv_msa_v1.0.tsv
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_iv_msa.csv -db db_iterations/XYZ_iv_v2.0_red.db -asp i -mod i -dialect msa -output_name conjugation/tables/conj_iv-i_msa_v1.0.tsv
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_iv_msa.csv -db db_iterations/XYZ_iv_v2.0_red.db -asp i -mod s -dialect msa -output_name conjugation/tables/conj_iv-s_msa_v1.0.tsv
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_iv_msa.csv -db db_iterations/XYZ_iv_v2.0_red.db -asp i -mod j -dialect msa -output_name conjugation/tables/conj_iv-j_msa_v1.0.tsv
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_cv_msa.csv -db db_iterations/XYZ_cv_v1.0_order-v4_red.db -asp c -dialect msa -output_name conjugation/tables/conj_cv_msa_v1.0.tsv

conj_tables_glf:
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_pv_glf.csv -db db_iterations/XYZ_glf_pv_v1.0.db -asp p -dialect glf -output_name conjugation/tables/conj_pv_glf_v1.0.tsv
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_iv_glf.csv -db db_iterations/XYZ_glf_iv_v1.0.db -asp i -dialect glf -output_name conjugation/tables/conj_iv_glf_v1.0.tsv
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_cv_glf.csv -db db_iterations/XYZ_glf_cv_v1.0.db -asp c -dialect glf -output_name conjugation/tables/conj_cv_glf_v1.0.tsv

conj_tables_egy:
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_pv_egy.csv -db db_iterations/XYZ_egy_pv_v1.0.db -asp p -dialect egy -output_name conjugation/tables/conj_pv_egy_v1.0.tsv
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_iv_egy.csv -db db_iterations/XYZ_egy_iv_v1.0.db -asp i -dialect egy -output_name conjugation/tables/conj_iv_egy_v1.0.tsv
	python generate_conj_table.py -paradigms conjugation/config_paradigms.json -repr_lemmas conjugation/repr_verbs/repr_verbs_cv_egy.csv -db db_iterations/XYZ_egy_cv_v1.0.db -asp c -dialect egy -output_name conjugation/tables/conj_cv_egy_v1.0.tsv

conj_tables: conj_tables_msa conj_tables_glf conj_tables_egy