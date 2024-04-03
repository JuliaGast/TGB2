'''
       Recurrency Baseline
	  
  File:     recurrencybaseline_predictor.py
  Authors:  Julia Gastinger (julia.gastinger@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) 2024, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
'''


import numpy as np
from collections import Counter
import time
import torch


class RecurrencyBaselinePredictor(object):
    def __init__(self, rels):
        """ init recurrency baseline predictor
        """
        self.rels = rels

    def apply_baselines(self, i, num_queries, test_data, all_data, window, basis_dict, num_nodes, 
                    num_rels, lmbda_psi, alpha, neg_samples, pos_sample, evaluator):
        """
        Apply baselines psi and xi (multiprocessing possible).

        Parameters:
            i (int): process number
            num_queries (int): minimum number of queries for each process
            test_data (np.array): test quadruples (only used in single-step prediction, depending on window specified);
                including inverse quadruples for subject prediction
            all_data (np.array): train valid and test quadruples (test only used in single-step prediction, depending 
                on window specified); including inverse quadruples  for subject prediction
            window: int, specifying which values from the past can be used for prediction. 0: all edges before the test 
            query timestamp are included. -2: multistep. all edges from train and validation set used. as long as they are 
            < first_test_query_ts. Int n > 0, all edges within n timestamps before the test query timestamp are included.
            basis_dict (dict): keys: rel_ids; specifies the predefined rules for each relation. 
                in our case: head rel = tail rel, confidence =1 for all rels in train/valid set
            score_func_psi (method): method to use for computing time decay for psi
            num_nodes (int): number of nodes in the dataset
            num_rels (int): number of relations in the dataset
            baselinexi_flag (boolean): True: use baselinexi, False: do not use baselinexi
            baselinepsi_flag (boolean): True: use baselinepsi, False: do not use baselinepsi
            lambda_psi (float): parameter for time decay function for baselinepsi. 0: no decay, >1 very steep decay
            alpha (float): parameter, weight to combine the scores from psi and xi. alpha*scores_psi + (1-alpha)*scores_xi
        Returns:
            logging_dict (dict): dict with one entry per test query (one per direction) key: string that desribes the query, 
            with xxx before the requested node, 
            values: list with two entries: [[tensor with one score per node], [np array with query_quadruple]]
            example: '14_0_xxx1_336': [tensor([1.8019e+01, ...9592e-05]), array([ 14,   0,   1...ype=int32)]
            scores_dict_eval (dict): dict  with one entry per test query (one per direction) key: str(test_qery), value: 
            tensor with scores, one score per node. example: [14, 0, 1, 336]':tensor([1.8019e+01,5.1101e+02,..., 0.0000e+0])
        """
        num_test_queries = len(test_data) - (i + 1) * num_queries
        if num_test_queries >= num_queries:
            test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
        else:
            test_queries_idx = range(i * num_queries, len(test_data))
        if num_queries >1:
            pos_sample = [pos_sample[i] for i in test_queries_idx]
            neg_samples = [neg_samples[i] for i in test_queries_idx]

        cur_ts = test_data[test_queries_idx[0]][3]
        first_test_query_ts = test_data[0][3]
        edges, all_data_ts = self.get_window_edges(all_data, cur_ts, window, first_test_query_ts) # get for the current 
                                # timestep all previous quadruples per relation that fullfill time constraints
        

        obj_dist = {}
        rel_obj_dist = {}
        rel_obj_dist_cur_ts, obj_dist_cur_ts = self.update_distributions(all_data_ts, edges, obj_dist, rel_obj_dist,num_rels, cur_ts)

        if len(all_data_ts) >0:
            sum_delta_t = self.update_delta_t(np.min(all_data_ts[:,3]), np.max(all_data_ts[:,3]), cur_ts, lmbda_psi)
            # sum_delta_t = np.max([sum_delta_t, 1e-15]) # to avoid division by zero

        it_start = time.time()
        logging_dict = {} # for logging
        scores_dict_eval = {}
        rel_ob_dist_scores = torch.zeros(num_nodes)
        predictions_xi=torch.zeros(num_nodes) 
        predictions_psi=torch.zeros(num_nodes)

        predictions_list = []
        hits_list = []
        perf_list = []
        for num_el, j in enumerate(test_queries_idx):       
            test_query = test_data[j]
            cands_dict = dict() 
            cands_dict_psi = dict() 
            # 1) update timestep and known triples
            if test_query[3] != cur_ts: # if we have a new timestep
                cur_ts = test_query[3]
                edges, all_data_ts = self.get_window_edges(all_data, cur_ts, window, first_test_query_ts) # get for the current timestep all previous quadruples per relation that fullfill time constraints
                # update the object and rel-object distritbutions to take into account what timesteps to use
                if window > -1: #otherwise: multistep, we do not need to update
                    rel_obj_dist_cur_ts, obj_dist_cur_ts = self.update_distributions(all_data_ts, edges, obj_dist_cur_ts, rel_obj_dist_cur_ts, num_rels, cur_ts)

                if len(all_data_ts) >0:
                    if window > -1: #otherwise: multistep, we do not need to update
                        sum_delta_t = self.update_delta_t(np.min(all_data_ts[:,3]), np.max(all_data_ts[:,3]), cur_ts, lmbda_psi)
                        # sum_delta_t = np.max([sum_delta_t, 1e-15]) # to avoid division by zero
                            
            #### BASELINE  PSI
            # 2) apply rules for relation of interest, if we have any

            if str(test_query[1]) in basis_dict: # do we have rules for the given relation?
                for rule in basis_dict[str(test_query[1])]:  # check all the rules that we have                    
                    walk_edges = self.match_body_relations(rule, edges, test_query[0]) 
                                        # Find quadruples that match the rule (starting from the test query subject)
                                        # Find edges whose subject match the query subject and the relation matches
                                        # the relation in the rule body. np array with [[sub, obj, ts]]
                    if 0 not in [len(x) for x in walk_edges]: # if we found at least one potential rule
                        
                        cands_dict_psi = self.get_candidates_psi(walk_edges[0][:,1:3], cur_ts, cands_dict, lmbda_psi, sum_delta_t)
                        if len(cands_dict_psi)>0:                
                            predictions_psi = self.create_scores_tensor(cands_dict_psi, num_nodes)

            #### BASELINE XI
            # obj_dist, rel_obj_dist            
            rel_ob_dist_scores = self.create_scores_tensor(rel_obj_dist_cur_ts[test_query[1]], num_nodes)
            predictions_xi = rel_ob_dist_scores

            # logging the scores in a format that is similar to other methods. needs a lot of memory.
    
            query_name, gt_test_query_ids = self.query_name_from_quadruple(test_query, num_rels)
            predictions_all = 1000*alpha*predictions_psi + 1000*(1-alpha)*predictions_xi 
            logging_dict[query_name] = [predictions_all, gt_test_query_ids]       
            scores_dict_eval[str(list(gt_test_query_ids))] = predictions_all
            
            # samples_of_interest = pos_sample[num_el] neg_samples[num_el]]
            predictions_of_interest_pos = predictions_all[pos_sample[num_el]].unsqueeze(0) #TODO WRONG INDEX
            predictions_of_interest_neg = predictions_all[neg_samples[num_el]] #TODO WRONG INDEX
            pboth = torch.cat((predictions_of_interest_pos, predictions_of_interest_neg))
            predictions_list.append(pboth)
            input_dict = {
                "y_pred_pos": np.array([predictions_of_interest_pos]),
                "y_pred_neg": np.array(predictions_of_interest_neg ),
                "eval_metric": ['mrr'], #TODO: also compute hits
            }
            perf_list.append(evaluator.eval(input_dict)['mrr'])
            hits_list.append(evaluator.eval(input_dict)['hits@10'])
            if pos_sample[num_el]  != test_query[2]:
                print(pos_sample[num_el], type(pos_sample[num_el]))
                print(test_query[2], type(test_query[2]))
            assert(pos_sample[num_el] == test_query[2])

        return logging_dict, scores_dict_eval, perf_list, hits_list
    

    def match_body_relations(self, rule, edges, test_query_sub):
        """
        for rules of length 1
        Find quadruples that match the rule (starting from the test query subject)
        Find edges whose subject match the query subject and the relation matches
        the relation in the rule body. 
        Memory-efficient implementation.

        modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
        shortened because we only have rules of length one 

        Parameters:
            rule (dict): rule from rules_dict
            edges (dict): edges for rule application
            test_query_sub (int): test query subject
        Returns:
            walk_edges (list of np.ndarrays): edges that could constitute rule walks
        """

        rels = rule["body_rels"]
        # Match query subject and first body relation
        try:
            rel_edges = edges[rels[0]]
            mask = rel_edges[:, 0] == test_query_sub
            new_edges = rel_edges[mask]
            walk_edges = [np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))]  # [sub, obj, ts]

        except KeyError:
            walk_edges = [[]]
        return walk_edges #subject object timestamp

    def score_delta(self, cands_ts, test_query_ts, lmbda):
        """ deta function to score a given candidate based on its distance to current timestep and based on param lambda
        Parameters:
            cands_ts (int): timestep of candidate(s)
            test_query_ts (int): timestep of current test quadruple
            lmbda (float): param to specify how steep decay is
        Returns:
            score (float): score for a given candicate
        """
        score = pow(2, lmbda * (cands_ts - test_query_ts))
        return score

    def get_window_edges(self, all_data, test_query_ts, window=-2, first_test_query_ts=0): #modified eval_paper_authors: added first_test_query_ts for validation set usage
        """
        modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
        introduce window -2 

        Get the edges in the data (for rule application) that occur in the specified time window.
        If window is 0, all edges before the test query timestamp are included.
        If window is -2, all edges from train and validation set are used. as long as they are < first_test_query_ts
        If window is an integer n > 0, all edges within n timestamps before the test query
        timestamp are included.

        Parameters:
            all_data (np.ndarray): complete dataset (train/valid/test)
            test_query_ts (np.ndarray): test query timestamp
            window (int): time window used for rule application
            first_test_query_ts (int): smallest timestamp from test set (eval_paper_authors)

        Returns:
            window_edges (dict): edges in the window for rule application
        """

        if window > 0:
            mask = (all_data[:, 3] < test_query_ts) * (
                all_data[:, 3] >= test_query_ts - window 
            )
            window_edges = self.quads_per_rel(all_data[mask]) # quadruples per relation that fullfill the time constraints 
        elif window == 0:
            mask = all_data[:, 3] < test_query_ts #!!! 
            window_edges = self.quads_per_rel(all_data[mask]) 
        elif window == -2: #modified eval_paper_authors: added this option
            mask = all_data[:, 3] < first_test_query_ts # all edges at timestep smaller then the test queries. meaning all from train and valid set
            window_edges = self.quads_per_rel(all_data[mask])  
        elif window == -200: #modified eval_paper_authors: added this option
            abswindow = 200
            mask = (all_data[:, 3] < first_test_query_ts) * (
                all_data[:, 3] >= first_test_query_ts - abswindow  # all edges at timestep smaller than the test queries - 200
            )
            window_edges = self.quads_per_rel(all_data[mask])
        all_data_ts = all_data[mask]
        return window_edges, all_data_ts


    def quads_per_rel(self, quads):
        """
        modified from Tlogic rule_application.py https://github.com/liu-yushan/TLogic/blob/main/mycode/rule_application.py
        Store all edges for each relation.

        Parameters:
            quads (np.ndarray): indices of quadruples

        Returns:
            edges (dict): edges for each relation
        """

        edges = dict()
        relations = list(set(quads[:, 1]))
        for rel in relations:
            edges[rel] = quads[quads[:, 1] == rel]
        return edges

    def get_candidates_psi(self, rule_walks, test_query_ts, cands_dict,lmbda, sum_delta_t):
        """
        Get answer candidates from the walks that follow the rule.
        Add the confidence of the rule that leads to these candidates.
        originally from TLogic https://github.com/liu-yushan/TLogic/blob/main/mycode/apply.py but heavily modified

        Parameters:
            rule_walks (np.array): rule walks np array with [[sub, obj]]
            test_query_ts (int): test query timestamp
            cands_dict (dict): candidates along with the confidences of the rules that generated these candidates
            score_func (function): function for calculating the candidate score
            lmbda (float): parameter to describe decay of the scoring function
            sum_delta_t: to be used in denominator of scoring fct
        Returns:
            cands_dict (dict): keys: candidates, values: score for the candidates  """

        cands = set(rule_walks[:,0]) 

        for cand in cands:
            cands_walks = rule_walks[rule_walks[:,0] == cand] 
            score = self.score_psi(cands_walks, test_query_ts, lmbda, sum_delta_t).astype(np.float64)
            cands_dict[cand] = score

        return cands_dict

    def update_delta_t(self, min_ts, max_ts, cur_ts, lmbda):
        """ compute denominator for scoring function psi_delta
        Patameters:
            min_ts (int): minimum available timestep
            max_ts (int): maximum available timestep
            cur_ts (int): current timestep
            lmbda (float): time decay parameter
        Returns:
            delta_all (float): sum(delta_t for all available timesteps between min_ts and max_ts)
        """
        timesteps = np.arange(min_ts, max_ts)
        now = np.ones(len(timesteps))*cur_ts
        delta_all = self.score_delta(timesteps, now, lmbda)
        delta_all = np.sum(delta_all)
        return delta_all

    def score_psi(self, cands_walks, test_query_ts, lmbda, sum_delta_t):
        """
        Calculate candidate score depending on the time difference.

        Parameters:
            cands_walks (np.array): rule walks np array with [[sub, obj]]
            test_query_ts (int): test query timestamp
            lmbda (float): rate of exponential distribution

        Returns:
            score (float): candidate score
        """

        all_cands_ts = cands_walks[:,1] #cands_walks["timestamp_0"].reset_index()["timestamp_0"]
        ts_series = np.ones(len(all_cands_ts))*test_query_ts 
        scores =  self.score_delta(all_cands_ts, ts_series, lmbda) # Score depending on time difference
        if sum_delta_t == 0:
            print(scores, "sum_delta_t is zero")
            print(all_cands_ts)
            score = np.sum(scores)
            # print(score)
        else:            
            score = np.sum(scores)/sum_delta_t

        return score   

    def update_distributions(self, learn_data_ts, ts_edges, obj_dist, 
                            rel_obj_dist, num_rels, cur_ts):
        """ update the distributions with more recent infos, if there is a more recent timestep available, depending on window parameter
        take into account scaling factor
        """
        obj_dist_cur_ts, rel_obj_dist_cur_ts= self.calculate_obj_distribution(learn_data_ts, ts_edges, num_rels) #, lmbda, cur_ts)
        return  rel_obj_dist_cur_ts, obj_dist_cur_ts
    
    def calculate_obj_distribution(self, learn_data, edges, num_rels):
        """
        Calculate the overall object distribution and the object distribution for each relation in the data.

        Parameters:
            learn_data (np.ndarray): data on which the rules should be learned
            edges (dict): edges from the data on which the rules should be learned

        Returns:
            obj_dist (dict): overall object distribution
            rel_obj_dist (dict): object distribution for each relation
        """
        obj_dist_scaled = {}

        rel_obj_dist = dict()
        rel_obj_dist_scaled = dict()
        for rel in range(num_rels):
            rel_obj_dist[rel] = {}
            rel_obj_dist_scaled[rel] = {}
        
        for rel in edges:
            objects = edges[rel][:, 2]
            dist = Counter(objects)
            for obj in dist:
                dist[obj] /= len(objects)
            rel_obj_dist_scaled[rel] = {k: v for k, v in dist.items()}

        return obj_dist_scaled, rel_obj_dist_scaled
    
    def update_delta_t(self, min_ts, max_ts, cur_ts, lmbda):
        """ compute denominator for scoring function psi_delta
        Patameters:
            min_ts (int): minimum available timestep
            max_ts (int): maximum available timestep
            cur_ts (int): current timestep
            lmbda (float): time decay parameter
        Returns:
            delta_all (float): sum(delta_t for all available timesteps between min_ts and max_ts)
        """
        timesteps = np.arange(min_ts, max_ts)
        now = np.ones(len(timesteps))*cur_ts
        delta_all = self.score_delta(timesteps, now, lmbda)
        delta_all = np.sum(delta_all)
        return delta_all


    def query_name_from_quadruple(self, quad, num_rels, plus_one_flag=False):
        """ get the query namefrom the given quadruple. if they do reverse prediction with nr*rel+rel_id then we undo it here
        :param quad: numpy array, len 4: [sub, rel, ob, ts]; if rel>num_rels-1: this means inverse prediction
        :param num_rels: [int] number of relations
        :param plus_one_flag: [Bool] if the number of relations for inverse predictions is one higher than expected - the case for timetraveler:self.quadruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3]]
        :return: 
        query_name [str]: name of the query, with xxx showing the entity of interest. e.g.'30_13_xxx_334' for 
            object prediction or 'xxx_13_18_334' for subject prediction
        test_query_ids [np array]: sub, rel, ob, ts (original rel id)
        """
        rel = quad[1]
        ts = quad[3]
        if rel > (num_rels-1): #wrong direction
            
            ob_pred = False
            if plus_one_flag == False:
                rel = rel - (num_rels) 
            else:
                rel = rel - (num_rels) -1 
            sub = quad[2]
            ob = quad[0]
        else:
            ob_pred = True
            sub = quad[0]
            ob = quad[2]      
        
        if ob_pred == True:
            query_name = str(sub) + '_' + str(rel) + '_' + 'xxx'+ str(ob) +'_' + str(ts)
        else:
            query_name = 'xxx'+ str(sub)+ '_' + str(rel) + '_' + str(ob) + '_'  + str(ts)
        
        test_query_ids = np.array([sub, rel, ob, ts])
        return query_name, test_query_ids

    def create_scores_tensor(self, predictions_dict, num_nodes, device=None):
        """ for given dict with key: node id, and value: score -> create a tensor with num_nodes entries, where the score 
        from dict is enetered at respective place, and all others are zeros.

        :returns: predictions  tensor with predicted scores, one per node; e.g. tensor([ 5.3042,  6....='cuda:0') torch.Size([23033])
        """
        predictions = torch.zeros(num_nodes, device=device)
        predictions.scatter_(0, torch.tensor(list(predictions_dict.keys())).long(), torch.tensor(list(predictions_dict.values())).float())
        return predictions