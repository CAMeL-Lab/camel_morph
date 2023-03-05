import numpy as np
import json
from copy import copy

from eval_utils import color, bold, underline


class DiacCombinationRow:
    RECALL_COLUMNS = ['no_intersect', 'exact_match',
                      'baseline_superset', 'system_superset',
                      'intersect']
    
    def __init__(self,
                 combination, match_total,
                 num_valid_lemmas, num_valid_feats,
                 diac_mat_baseline, diac_mat_system,
                 system_only_mat, baseline_only_mat,
                 no_intersect_mat) -> None:
        self.combination = combination
        self.diac_mat_baseline = diac_mat_baseline
        self.diac_mat_system = diac_mat_system
        self.system_only_mat = system_only_mat
        self.baseline_only_mat = baseline_only_mat
        self.no_intersect_mat = no_intersect_mat

        self.num_diac_baseline = combination['baseline']
        self.num_diac_system = combination['system']
        self.slots_color = combination.get('slots_color')
        self.diacs_color = combination.get('diacs_color')
        self.lemmas_feats_color = combination.get('lemmas_feats_color')
        
        match_comb_mask = self.combination['match_comb_mask']

        self.match_comb_mask_sum = int(np.sum(match_comb_mask))
        if self.match_comb_mask_sum == 0:
            return
        else:
            self.match_comb_mask_perc = self.match_comb_mask_sum / match_total
            self.num_lemmas_match_sum = int(np.sum(np.any(match_comb_mask, axis=1)))
            self.num_feats_match_sum = int(np.sum(np.any(match_comb_mask, axis=0)))
            self.num_lemmas_match_perc = self.num_lemmas_match_sum / num_valid_lemmas
            self.num_feats_match_perc = self.num_feats_match_sum / num_valid_feats
            self.compute_sums(match_comb_mask)
            self.compute_distributions()
            self.compute_average_recall()
            self.recall_avg_str()
            self.recall_total_dist_str()


    def compute_sums(self, match_comb_mask):
        self.no_intersect_mask_sum_slots, self.no_intersect_mask_sum_diacs = \
            self._no_intersect_mask_sum(match_comb_mask)
        self.exact_match_mask_sum_slots, self.exact_match_mask_sum_diacs = \
            self._exact_match_mask_sum(match_comb_mask)
        self.system_superset_mask_sum_slots, self.system_superset_mask_sum_diacs = \
            self._system_superset_mask_sum(match_comb_mask)
        self.baseline_superset_mask_sum_slots, self.baseline_superset_mask_sum_diacs = \
            self._baseline_superset_mask_sum(match_comb_mask)
        self.intersect_mask_sum_slots, self.intersect_mask_sum_diacs = \
            self._intersect_mask_sum(match_comb_mask)
        
        assert self.no_intersect_mask_sum_slots == self.no_intersect_mask_sum_diacs

        if all(type(self.combination[x]) is int for x in ['system', 'baseline']):
            if self.combination['baseline'] == self.combination['system']:
                assert self.exact_match_mask_sum_diacs == self.exact_match_mask_sum_slots == \
                       self.exact_match_mask_sum_slots * self.combination['baseline'] == \
                       self.exact_match_mask_sum_slots * self.combination['system']
                assert self.system_superset_mask_sum_diacs == \
                    self.baseline_superset_mask_sum_diacs == 0
            else:
                assert self.exact_match_mask_sum_diacs == 0
        
        self.total_diacs_baseline = int(np.sum(self.diac_mat_baseline[match_comb_mask]))
        self.total_diacs_system = int(np.sum(self.diac_mat_system[match_comb_mask]))

    
    def compute_distributions(self):
        self.recall_diacs_total_x_y = sum(getattr(self, f'{col_name}_mask_sum_diacs')
                                      for col_name in DiacCombinationRow.RECALL_COLUMNS)
        self.recall_slots_total_x_y = sum(getattr(self, f'{col_name}_mask_sum_slots')
                                      for col_name in DiacCombinationRow.RECALL_COLUMNS)
        assert self.recall_slots_total_x_y == self.match_comb_mask_sum
        if all(type(self.combination[x]) is int for x in ['system', 'baseline']):
            assert self.recall_slots_total_x_y * self.combination['baseline'] == self.total_diacs_baseline
        
        self.recall_diacs_x_y_dist = [
            (getattr(self, f'{col_name}_mask_sum_diacs') / self.recall_diacs_total_x_y)
            if self.recall_diacs_total_x_y else getattr(self, f'{col_name}_mask_sum_diacs')
            for col_name in DiacCombinationRow.RECALL_COLUMNS]
        self.recall_slots_x_y_dist = [
            getattr(self, f'{col_name}_mask_sum_slots') / self.recall_slots_total_x_y
            for col_name in DiacCombinationRow.RECALL_COLUMNS]

    
    def compute_average_recall(self):
        for system_ in ['baseline', 'system']:
            total_diacs_system_ = getattr(self, f'total_diacs_{system_}')
            if total_diacs_system_ != 0:
                recall_diacs_system_ = self.recall_diacs_total_x_y - self.no_intersect_mask_sum_diacs
                recall_diacs_system_ /= total_diacs_system_
                recall_slots_system_ = self.recall_slots_total_x_y - self.no_intersect_mask_sum_slots
                recall_slots_system_ /= self.match_comb_mask_sum
                recall_diacs_baseline_str = f'{recall_diacs_system_:.1%}'
                recall_slots_baseline_str = f'{recall_slots_system_:.1%}'
            else:
                recall_diacs_system_, recall_slots_system_ = 'N/A', 'N/A'
                recall_diacs_baseline_str, recall_slots_baseline_str = 'N/A', 'N/A'

            setattr(self, f'recall_diacs_{system_}', recall_diacs_system_)
            setattr(self, f'recall_slots_{system_}', recall_slots_system_)
            setattr(self, f'recall_diacs_{system_}_str', recall_diacs_baseline_str)
            setattr(self, f'recall_slots_{system_}_str', recall_slots_baseline_str)


    def _no_intersect_mask_sum(self, match_comb_mask):
        no_intersect_mask_slots = int(np.sum(self.no_intersect_mat[match_comb_mask]))
        no_intersect_mask_diacs = no_intersect_mask_slots
        return no_intersect_mask_slots, no_intersect_mask_diacs
    
    def _exact_match_mask_sum(self, match_comb_mask):
        exact_match_mask = ((self.baseline_only_mat == 0) &
                            (self.system_only_mat == 0) &
                            (self.no_intersect_mat == False) &
                            match_comb_mask)
        exact_match_mask_sum_slots = int(np.sum(exact_match_mask))
        exact_match_mask_sum_diacs = int(np.sum(self.diac_mat_system[exact_match_mask]))
        
        return exact_match_mask_sum_slots, exact_match_mask_sum_diacs
    
    def _system_superset_mask_sum(self, match_comb_mask):
        system_superset_mask = ((self.system_only_mat != 0) &
                                (self.baseline_only_mat == 0) &
                                match_comb_mask)
        system_superset_mask_sum_slots = int(np.sum(system_superset_mask))
        if all(type(self.combination[x]) is int for x in ['system', 'baseline']) and \
           self.combination['baseline'] >= self.combination['system']:
            assert system_superset_mask_sum_slots == 0
        system_superset_mask_sum_diacs = np.minimum(self.diac_mat_system[system_superset_mask],
                                                    self.diac_mat_baseline[system_superset_mask])
        system_superset_mask_sum_diacs = int(np.sum(system_superset_mask_sum_diacs))
        
        return system_superset_mask_sum_slots, system_superset_mask_sum_diacs
    
    def _baseline_superset_mask_sum(self, match_comb_mask):
        baseline_superset_mask = ((self.system_only_mat == 0) &
                                  (self.baseline_only_mat != 0) &
                                  match_comb_mask)
        baseline_superset_mask_sum_slots = int(np.sum(baseline_superset_mask))
        if all(type(self.combination[x]) is int for x in ['system', 'baseline']) and \
           self.combination['system'] >= self.combination['baseline']:
            assert baseline_superset_mask_sum_slots == 0
        baseline_superset_mask_sum_diacs = int(
            np.sum(self.diac_mat_system[baseline_superset_mask]))
        
        return baseline_superset_mask_sum_slots, baseline_superset_mask_sum_diacs
    
    def _intersect_mask_sum(self, match_comb_mask):
        intersect_mask = ((self.baseline_only_mat != 0) &
                          (self.system_only_mat != 0) &
                          match_comb_mask)
        intersect_mask_sum_slots = int(np.sum(intersect_mask))
        assert np.sum(self.system_only_mat[intersect_mask]) == \
               np.sum(self.baseline_only_mat[intersect_mask])
        intersect_mask_sum_diacs = int(np.sum(self.system_only_mat[intersect_mask]))

        return intersect_mask_sum_slots, intersect_mask_sum_diacs
    

    def val_and_perc_str(self, attr, color_=None):
        sum_, perc_ = getattr(self, f'{attr}_sum'), getattr(self, f'{attr}_perc')
        sum_, perc_= f'{sum_:,}', f'{perc_:.1%}'
        if color_ is not None:
            setattr(self, f'{attr}_str', bold(color(sum_, color_)) + '\n' + bold(perc_))
        else:
            setattr(self, f'{attr}_str', sum_ + '\n' + perc_)

    def recall_avg_str(self):
        self.recall_diac_highest_index = np.array(self.recall_diacs_x_y_dist).argmax()
        self.recall_slot_highest_index = np.array(self.recall_slots_x_y_dist).argmax()
        diacs_color = self.diacs_color if self.diacs_color is not None else 'green'
        slots_color = self.slots_color if self.slots_color is not None else 'cyan'
        for i, col in enumerate(DiacCombinationRow.RECALL_COLUMNS):
            recall_col_diacs = self.recall_diacs_x_y_dist[i]
            recall_col_slots = self.recall_slots_x_y_dist[i]
            if i != self.recall_diac_highest_index:
                recall_col_diacs_str = f'{recall_col_diacs:.1%}'
            else:
                recall_col_diacs_str = bold(color(f'{recall_col_diacs:.1%}', diacs_color))
            if recall_col_diacs != recall_col_slots:
                if i != self.recall_slot_highest_index:
                    recall_col_slots_str = f'\n{recall_col_slots:.1%}'
                else:
                    recall_col_slots_str = '\n' + bold(color(f'{recall_col_slots:.1%}', slots_color))
            else:
                recall_col_slots_str = ''
            
            setattr(self, f'recall_{col}_str', recall_col_diacs_str + recall_col_slots_str)
    
    def recall_total_dist_str(self):
        for system_ in ['baseline', 'system']:
            recall_diacs_system_str = getattr(self, f'recall_diacs_{system_}_str')
            recall_slots_system_str = getattr(self, f'recall_slots_{system_}_str')
            if recall_diacs_system_str != recall_slots_system_str:
                recall_slots_system_str = '\n' + bold(color(recall_slots_system_str, 'blue'))
            else:
                recall_slots_system_str = ''
            recall_system_str = bold(color(recall_diacs_system_str, 'blue')) + recall_slots_system_str
            setattr(self, f'recall_{system_}_str', recall_system_str)


class SpecificFeatStatsRow:
    def __init__(self,
                 row_info=None,
                 system_info=None) -> None:
        if system_info is not None:
            self.system_info = system_info
            self.num_diac_baseline = system_info.combination[0]
            self.num_diac_system = system_info.combination[1]

        if row_info is not None:
            if type(row_info) is tuple:
                # Group 1
                self.feat = row_info[0]
                self.value = row_info[1]
                self.pos = row_info[2]
                self.feat_value = f'{self.pos.upper()}(' + f'{self.feat}:{self.value}' + ')'
                self.compute_values_group_1()
            else:
                # Group 2
                if row_info['feats_dict'] != 'UNSORTED (Total)':
                    self.feat_value = ' '.join(
                        sorted(f'{f}:{v}' for f, v in row_info['feats_dict'].items()))
                else:
                    self.feat_value = row_info['feats_dict']
                self.explanation = row_info['explanation']
                self.compute_values_group_2()
        
            self.compute_values_str()


    def compute_values_group_1(self):
        feat_value_indexes_ = np.array(
            [i for i, feat_comb in enumerate(self.system_info.index2analysis)
             if self.value == feat_comb[self.system_info.feat2index[self.feat]]])
        group_1_mask = self.system_info.diac_mat[:, feat_value_indexes_] != 0
    
        match_comb_indexes = np.where(group_1_mask)
        match_comb_indexes = ([match_comb_indexes[0][0]],
                              [feat_value_indexes_[match_comb_indexes[1][0]]])
        self.match_comb_indexes = match_comb_indexes
        
        self.num_slots_sum = int(np.sum(group_1_mask))
        self.num_slots_perc = self.num_slots_sum / self.system_info.group_1_slots_sum
        self.num_lemmas_valid_sum = int(np.sum(np.any(group_1_mask, axis=1)))
        self.num_lemmas_valid_perc = self.num_lemmas_valid_sum / self.system_info.total_lemmas
        self.num_feats_valid_sum = len(feat_value_indexes_)
        self.num_feats_valid_perc = self.num_feats_valid_sum / self.system_info.total_feats


    def compute_values_group_2(self):
        feat_combs_indexes_ = np.array([
            self.system_info.analysis2index[feats]
            for feats in self.system_info.group_2_categorization[self.feat_value]])
        group_2_mask = self.system_info.diac_mat[:, feat_combs_indexes_] != 0
    
        match_comb_indexes = np.where(group_2_mask)
        match_comb_indexes = ([match_comb_indexes[0][0]], [feat_combs_indexes_[0]])
        self.match_comb_indexes = match_comb_indexes
        
        self.num_slots_sum = int(np.sum(group_2_mask))
        self.num_slots_perc = self.num_slots_sum / self.system_info.group_2_slots_sum
        self.num_lemmas_valid_sum = int(np.sum(np.any(group_2_mask, axis=1)))
        self.num_lemmas_valid_perc = self.num_lemmas_valid_sum / self.system_info.total_lemmas
        self.num_feats_valid_sum = len(self.system_info.group_2_categorization[self.feat_value])
        self.num_feats_valid_perc = self.num_feats_valid_sum / self.system_info.group_2_feat_combs_sum


    def compute_values_str(self):
        num_slots_sum_str = f'{self.num_slots_sum:,}'
        num_slots_perc_str = f'({self.num_slots_perc:.1%})'
        self.num_slots_str = num_slots_sum_str + '\n' + num_slots_perc_str
        num_lemmas_valid_sum_str = f'{self.num_lemmas_valid_sum:,}'
        num_lemmas_valid_perc_str = f'{self.num_lemmas_valid_perc:.1%}'
        self.num_lemmas_valid_str = num_lemmas_valid_sum_str + '\n' + num_lemmas_valid_perc_str
        num_feats_valid_sum_str = f'{self.num_feats_valid_sum:,}'
        num_feats_valid_perc_str = f'{self.num_feats_valid_perc:.1%}'
        self.num_feats_valid_str = num_feats_valid_sum_str + '\n' + num_feats_valid_perc_str

    
    @staticmethod
    def compute_total_row_group_2(rows):
        for row in rows:
            if row.feat_value == 'UNSORTED (Total)':
                num_slots_sum_unsorted = row.num_slots_sum
                break
        else:
            num_slots_sum_unsorted = 0
        
        sorted_total_row = SpecificFeatStatsRow()
        sorted_total_row.num_diac_baseline = rows[0].num_diac_baseline
        sorted_total_row.num_diac_system = rows[0].num_diac_system
        sorted_total_row.feat_value = 'SORTED (Total)'
        group_2_slots_sum = rows[0].system_info.group_2_slots_sum
        sorted_total_row.num_slots_sum = group_2_slots_sum - num_slots_sum_unsorted
        sorted_total_row.num_slots_str = (
            f'{group_2_slots_sum - num_slots_sum_unsorted:,}' + '\n' +
            f'({(group_2_slots_sum - num_slots_sum_unsorted)/group_2_slots_sum:.1%})')
        sorted_total_row.num_lemmas_valid_str = '-'
        sorted_total_row.num_feats_valid_str = '-'
        sorted_total_row.example_str = '-'
        sorted_total_row.explanation = '-'

        total_row = copy(sorted_total_row)
        total_row.feat_value = bold(color('Total', 'orange'))
        total_row.num_slots_sum = rows[0].system_info.group_2_slots_sum
        total_row.num_slots_str = (
            bold(color(f'{total_row.num_slots_sum:,}', 'orange')) + '\n' + bold(f'({1:.1%})'))


        return [sorted_total_row, total_row]
        
        
class SystemInfoUnsharedFeats:
    def __init__(self, feat_subcombs_path, combination, system_, MATRICES, feat_value_pairs, feat2index) -> None:
        self.feat_subcombs_path = feat_subcombs_path
        self.combination = combination
        self.num_diac_baseline = combination[0]
        self.num_diac_system = combination[1]
        self.system_ = system_

        if feat_subcombs_path is not None:
            with open(feat_subcombs_path) as f:
                feat_subcombs = json.load(f)['unshared']
            self.feat_subcombs = feat_subcombs
        else:
            self.feat_subcombs = []
        
        self.diac_mat = MATRICES[f'{system_}_only'][f'diac_mat_{system_}']
        self.index2lemmas_pos = MATRICES[f'{system_}_only']['index2lemmas_pos']
        self.index2analysis = MATRICES[f'{system_}_only']['index2analysis']
        self.analysis2index = MATRICES[f'{system_}_only']['analysis2index']
        self.feat2index = feat2index
        self.group_1_pairs = feat_value_pairs[f'{system_}_only']
        self.group_2_categorization = self.get_group2categorization()
        self.group_2_indexes = np.array([
            self.analysis2index[feat_comb]
            for feat_combs in self.group_2_categorization.values()
            for feat_comb in feat_combs])
        self.group_2_mask = self.diac_mat[:, self.group_2_indexes] != 0
        self.group_2_slots_sum = int(np.sum(self.group_2_mask))
        self.group_2_feat_combs_sum = sum(
            len(feats) for feats in self.group_2_categorization.values())
        
        self.total_lemmas, self.total_feats = self.diac_mat.shape[0], self.diac_mat.shape[1]
        self.unshared_total = self.total_lemmas * self.total_feats
        self.unshared_mask = self.diac_mat != 0
        self.unshared_slots_sum = np.sum(self.unshared_mask)
        self.group_1_feat_combs_sum = self.total_feats - self.group_2_feat_combs_sum
        self.group_1_slots_sum = self.unshared_slots_sum - self.group_2_slots_sum
        self.num_lemmas_valid = np.sum(np.any(self.unshared_mask, axis=1))
        self.num_feats_valid = np.sum(np.any(self.unshared_mask, axis=0))

    
    def compute_values_group_1_str(self):
        self.num_diac_baseline_str = str(self.num_diac_baseline)
        self.num_diac_system_str = str(self.num_diac_system) + '\n(group 1)'
        self.num_slots_str = (f'{self.group_1_slots_sum:,}' + '\n' +
                                f'({self.group_1_slots_sum/self.unshared_slots_sum:.1%})')
        self.num_lemmas_valid_str = '-'
        self.num_feats_valid_str = '-'
        self.example_str = '-'
        self.feat_value_pairs_str = self.get_feat_value_pairs_str()

    def compute_values_group_2_str(self):
        self.num_diac_baseline_str = str(self.num_diac_baseline)
        self.num_diac_system_str = str(self.num_diac_system) + '\n(group 2)'
        self.num_slots_str = (f'{self.group_2_slots_sum:,}' + '\n' +
                                      f'({self.group_2_slots_sum/self.unshared_slots_sum:.1%})')
        self.num_lemmas_valid_str = '-'
        self.num_feats_valid_str = '-'
        self.example_str = '-'
        self.feat_value_pairs_str = '(same as above)'

    def compute_values_all_str(self):
        self.num_diac_baseline_str = str(self.num_diac_baseline)
        self.num_diac_system_str = str(self.num_diac_system) + '\n(all)'
        self.num_slots_str = (f'{self.unshared_slots_sum:,}' + '\n' +
                            f'({self.unshared_slots_sum/self.unshared_total:.1%})')
        self.num_lemmas_valid_str = (f'{self.num_lemmas_valid:,}' + '\n' +
                                     f'({self.num_lemmas_valid/self.total_lemmas:.1%})')
        self.num_feats_valid_str = (f'{self.num_feats_valid:,}' + '\n' +
                                    f'({self.num_feats_valid/self.total_feats:.1%})')
        self.feat_value_pairs_str = '-'

    def compute_values_0_0_str(self):
        self.num_diac_baseline_str = f'0\n({self.system_})'
        self.num_diac_system_str = f'0\n({self.system_})'
        self.num_slots_str = (f'{self.unshared_total - self.unshared_slots_sum:,}' + '\n' +
                            f'({(self.unshared_total - self.unshared_slots_sum)/self.unshared_total:.1%})')
        self.num_lemmas_valid_str = '-'
        self.num_feats_valid_str = '-'
        self.feat_value_pairs_str = '-'


    def get_feat_value_pairs_str(self):
        pos2feat_value_pairs = {}
        for feat, value, pos in self.group_1_pairs:
            pos2feat_value_pairs.setdefault(pos, set()).add(f'{feat}:{value}')
        
        feat_value_pairs_str = '; '.join(pos.upper() + '(' + ' '.join(sorted(feat_value)) + ')'
                                    for pos, feat_value in pos2feat_value_pairs.items())
        return feat_value_pairs_str

    
    def get_group2categorization(self):
        # Contains for each feat:value the analyses (from possible combinations) that contain it
        feat_value2queries = {}
        for feat, value, pos in self.group_1_pairs:
            #NOTE: not using POS, might be problematic for OTHER
            for i, feats in enumerate(self.index2analysis):
                if feats[self.feat2index[feat]] == value:
                    feat_value2queries.setdefault((feat, value), []).append(i)
        # Contains for each feat:value the number of slots it occupies
        feat_value2slots = {}
        for (feat, value), queries in feat_value2queries.items():
            queries = np.array(queries)
            queries_valie_slots = self.diac_mat[:, queries]
            feat_value2slots[(feat, value)] = np.sum(queries_valie_slots)
        # Contains all the feat_combs in which the feat:value pairs participate
        feat_combs_group_1_indexes = set.union(*map(set, feat_value2queries.values()))
        # Contains all the feat_combs which do not have any of these feat:value pairs
        # but which are still unique to that system. These might either be invalid OR
        # mappable to one(s) in that of another system.
        feat_combs_group_2_indexes = [i for i in range(len(self.index2analysis))
                                        if i not in feat_combs_group_1_indexes]
        feat_combs_group_2 = [feats for i, feats in enumerate(self.index2analysis)
                                        if i in feat_combs_group_2_indexes]
        group_2_categorization = {}
        # feat_subcombs might be empty since it is filled manually
        invalid = False
        for feat_comb in feat_combs_group_2:
            for info in self.feat_subcombs:
                feats_dict = info['feats_dict']
                invalid = True
                for f, v_rule in feats_dict.items():
                    v = feat_comb[self.feat2index[f]]
                    match = False
                    for v_rule_ in v_rule.split('+'):
                        match = match or (v_rule_[0] == '!' and v != v_rule_[1:] or
                                        v_rule_[0] != '!' and v == v_rule_)
                    invalid = invalid and match
                if invalid:
                    break
            if invalid:
                group_2_categorization.setdefault(
                    ' '.join(sorted(f'{f}:{v}' for f, v in info['feats_dict'].items())), []).append(feat_comb)
            else:
                group_2_categorization.setdefault('UNSORTED (Total)', []).append(feat_comb)
        assert sum(len(feats) for feats in group_2_categorization.values()) == len(feat_combs_group_2)
        
        return group_2_categorization