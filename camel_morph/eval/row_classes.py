import numpy as np

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
                 row_info,
                 system_info,
                 feat2index) -> None:
        self.num_diac_baseline = system_info['combination'][0]
        self.num_diac_system = system_info['combination'][1]

        self.diac_mat_ = system_info['diac_mat_']
        self.index2analysis = system_info['index2analysis']
        self.analysis2index = system_info['analysis2index']
        self.index2lemmas_pos = system_info['index2lemmas_pos']
        self.group_2_categorization = system_info['group_2_categorization']
        self.total_lemmas = system_info['total_lemmas']
        self.total_feats = system_info['total_feats']
        self.unshared_sum = system_info['unshared_sum']
        self.total_group_2_sum = system_info['total_group_2_sum']
        self.feat2index = feat2index

        self.total_feat_combs_group_2 = sum(len(feats) for feats in self.group_2_categorization.values())
        self.total_feat_combs = self.total_feats - self.total_feat_combs_group_2
        self.total_group_1_sum = self.unshared_sum - self.total_group_2_sum

        if type(row_info) is tuple:
            # Group 1
            self.feat = row_info[0]
            self.value = row_info[1]
            self.pos = row_info[2]
            self.feat_value = f'{self.pos.upper()}(' + f'{self.feat}:{self.value}' + ')'
            self.compute_values_group_1()
        else:
            # Group 2
            self.feat_combs_indexes = system_info['feat_combs_indexes']
            self.feat_value = ' '.join(sorted(f'{f}:{v}' for f, v in row_info['feats_dict'].items()))
            self.explanation = row_info['explanation']
            self.compute_values_group_2()
        
        self.compute_values_str()


    def compute_values_group_1(self):
        feat_value_indexes_ = np.array([i for i, feat_comb in enumerate(self.index2analysis)
                                        if self.value == feat_comb[self.feat2index[self.feat]]])
        group_1_mask = self.diac_mat_[:, feat_value_indexes_] != 0
    
        match_comb_indexes = np.where(group_1_mask)
        match_comb_indexes = ([match_comb_indexes[0][0]], [feat_value_indexes_[match_comb_indexes[1][0]]])
        self.match_comb_indexes = match_comb_indexes
        
        self.num_slots_sum = int(np.sum(group_1_mask))
        self.num_slots_perc = self.num_slots_sum / self.total_group_1_sum
        self.num_lemmas_valid_sum = int(np.sum(np.any(group_1_mask, axis=1)))
        self.num_lemmas_valid_perc = self.num_lemmas_valid_sum / self.total_lemmas
        self.num_feats_valid_sum = len(feat_value_indexes_)
        self.num_feats_valid_perc = self.num_feats_valid_sum / self.total_feats


    def compute_values_group_2(self):
        feat_combs_indexes_ = np.array([self.analysis2index[feats]
                                        for feats in self.group_2_categorization[self.feat_value]])
        group_2_mask = self.diac_mat_[:, feat_combs_indexes_] != 0
    
        match_comb_indexes = np.where(group_2_mask)
        match_comb_indexes = ([match_comb_indexes[0][0]], [feat_combs_indexes_[0]])
        self.match_comb_indexes = match_comb_indexes
        
        self.num_slots_sum = int(np.sum(group_2_mask))
        self.num_slots_perc = self.num_slots_sum / self.total_group_2_sum
        self.num_lemmas_valid_sum = int(np.sum(np.any(group_2_mask, axis=1)))
        self.num_lemmas_valid_perc = self.num_lemmas_valid_sum / self.total_lemmas
        self.num_feats_valid_sum = len(self.group_2_categorization[self.feat_value])
        self.num_feats_valid_perc = self.num_feats_valid_sum / self.total_feat_combs_group_2


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