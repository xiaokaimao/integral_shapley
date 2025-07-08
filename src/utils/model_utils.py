def return_model(mode, **kwargs):
    '''
    Define a model to be used in computation of data values
    '''
    import inspect
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import MultinomialNB
    
    if inspect.isclass(mode):
        assert getattr(mode, 'fit', None) is not None, 'Custom model family should have a fit() method'
        model = mode(**kwargs)
    elif mode=='logistic':
        model = LogisticRegression(random_state=666)
    elif mode=='linear':
        n_jobs = kwargs.get('n_jobs', -1)
        model = LinearRegression(n_jobs=n_jobs)
    elif mode=='ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=666)
    elif mode=='Tree':
        model = DecisionTreeClassifier(random_state=666)
    elif mode=='RandomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='GB':
        n_estimators = kwargs.get('n_estimators', 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='AdaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif mode=='SVC':
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 0.05)
        print(f'C: {C}')
        max_iter = kwargs.get('max_iter', 5000)
        model = SVC(kernel=kernel, C=1, random_state=666, probability=True)
    elif mode=='LinearSVC':
        C = kwargs.get('C', 1)
        max_iter = kwargs.get('max_iter', 5000)
        model = SVC(kernel='linear', C=1, random_state=666, probability=True)
    elif mode=='GP':
        model = GaussianProcessClassifier(random_state=666)
    elif mode=='KNN':
        n_neighbors = kwargs.get('n_neighbors', 5)
        n_jobs = kwargs.get('n_jobs', -1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    elif mode=='NB':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid mode!")
    return model