from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

def evaluate(features):
    do_use = 0
    use = None
    sim_thres = 0
    # evaluate with USE

    if do_use == 1:
        cache_path = ''
        import tensorflow as tf
        import tensorflow_hub as hub
    
        class USE(object):
            def __init__(self, cache_path):
                super(USE, self).__init__()

                self.embed = hub.Module(cache_path)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session()
                self.build_graph()
                self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

            def build_graph(self):
                self.sts_input1 = tf.placeholder(tf.string, shape=(None))
                self.sts_input2 = tf.placeholder(tf.string, shape=(None))

                sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
                sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
                self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
                clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
                self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

            def semantic_sim(self, sents1, sents2):
                sents1 = [s.lower() for s in sents1]
                sents2 = [s.lower() for s in sents2]
                scores = self.sess.run(
                    [self.sim_scores],
                    feed_dict={
                        self.sts_input1: sents1,
                        self.sts_input2: sents2,
                    })
                return scores[0]

            use = USE(cache_path)


    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    y_true = []
    y_after_attack = []
    y_before_attack = []
    for feat in features:
        y_true.append(feat.label)
        y_after_attack.append(feat.after_attack_label)
        y_before_attack.append(feat.pred_label)
        if feat.success > 2:

            if do_use == 1:
                sim = float(use.semantic_sim([feat.code1], [feat.final_adverse]))
                if sim < sim_thres:
                    continue
            
            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.code1.split(' '))

            if feat.success == 3:
                origin_success += 1

        total += 1

    suc = float(acc / total)

    if acc != 0:
        query = float(total_q / acc)
    else:
        query = 0
    if total_word != 0:
        change_rate = float(total_change / total_word)
    else:
        change_rate = 0

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print("######## Before Attack ############")
    print('Accuracy:', accuracy_score(y_true, y_before_attack))
    print('F1 score:', f1_score(y_true, y_before_attack))
    print('Recall:', recall_score(y_true, y_before_attack))
    print('Precision:', precision_score(y_true, y_before_attack))
    print('Clasification report:\n', classification_report(y_true,y_before_attack))
    print('Confussion matrix:\n',confusion_matrix(y_true, y_before_attack))

    print("######## After Attack ############")
    print('Accuracy:', accuracy_score(y_true, y_after_attack))
    print('F1 score:', f1_score(y_true, y_after_attack))
    print('Recall:', recall_score(y_true, y_after_attack))
    print('Precision:', precision_score(y_true, y_after_attack))
    print('Clasification report:\n', classification_report(y_true,y_after_attack))
    print('Confussion matrix:\n',confusion_matrix(y_true, y_after_attack))

    print("#################")

    print('acc/aft-atk-acc {:.6f}/ {:.6f}, query-num {:.4f}, change-rate {:.4f}'.format(origin_acc, after_atk, query, change_rate))

