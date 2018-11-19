import pandas as pd
import numpy as np
from sklearn.cross_validation import _BaseKFold

def getTrainTimes(t1,
                  testTimes):
    """SNIPPET 7.1 PURGING OBSERVATION IN THE TRAINING SET
    Given testTimes, find the times of the training observations.
    —t1.index: Time when the observation started.
    —t1.value: Time when the observation ended.
    —testTimes: Times of testing observations.
    """
    trn = t1.copy(deep=True)
    for i, j in testTimes.iteritems():
        df0 = trn[(i <= trn.index) & (trn.index <= j)
                  ].index  # train starts within test
        df1 = trn[(i <= trn) & (trn <= j)].index  # train ends within test
        df2 = trn[(trn.index <= i) & (j <= trn)].index  # train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn

def getEmbargoTimes(times,
                    pctEmbargo):
    """SNIPPET 7.2 EMBARGO ON TRAINING OBSERVATIONS
    Get embargo time for each bar
    """
    step = int(times.shape[0]*pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = mbrg.append(pd.Series(times[-1], index=times[-step:]))
    return mbrg

class PurgedKFold(_BaseKFold):
    """SNIPPET 7.3 CROSS-VALIDATION CLASS WHEN OBSERVATIONS OVERLAP
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """
    def __init__(self, n, n_folds=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(
            n, n_folds, shuffle=False, random_state=None)  
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0]*self.pctEmbargo)     
        test_starts = [(i[0], i[-1]+1) for i in
                       np.array_split(np.arange(X.shape[0]), self.n_folds)]   
        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx+mbrg:]))
            yield train_indices, test_indices

def cvScore(clf,
            X,
            y,
            sample_weight,
            scoring='neg_log_loss',
            t1=None,
            cv=None,
            cvGen=None,
            pctEmbargo=None):
    """SNIPPET 7.4 USING THE PurgedKFold CLASS
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score
    #from clfSequential import PurgedKFold
    if cvGen is None:
        cvGen = PurgedKFold(n = X.shape[0], n_folds=cv, t1=t1,
                            pctEmbargo=pctEmbargo)  # purged
    score = []
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        print(f'Fold: {i}')
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train],
                      sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -1 * \
                log_loss(
                    y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(
                y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)
