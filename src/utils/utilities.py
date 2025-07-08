# utilities.py
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

#########################################
# 1. 基于 Hilbert 空间（RKHS）的效用函数
#########################################


def rbf_kernel(X, Y, gamma=0.1):
    """
    RBF 核函数，计算两个样本矩阵 X 和 Y 之间的核矩阵。
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    K = np.exp(-gamma * (X_norm + Y_norm - 2 * np.dot(X, Y.T)))
    return K

def rkhs_inner_product_multiclass(model1, model2, kernel_func, gamma=0.1, aggregation='average'):
    """
    针对多分类 SVM 模型：
    对每个二分类器（即 dual_coef_ 的每一行）分别计算 RKHS 内积，
    然后按指定的方式（默认为平均）汇总各个二分类器的内积。
    内积计算公式: inner_i = coeffs1_i^T * K * coeffs2_i,
    其中 K 是模型1和模型2支持向量之间的核矩阵。
    """
    sv1 = model1.support_vectors_
    sv2 = model2.support_vectors_
    K = kernel_func(sv1, sv2, gamma=gamma)

    # dual_coef_ 对应各个二分类器
    dual1 = model1.dual_coef_  # shape: (n_binary, n_sv1)
    dual2 = model2.dual_coef_  # shape: (n_binary, n_sv2)

    n_classifiers = dual1.shape[0]
    inner_products = []
    for i in range(n_classifiers):
        coeffs1 = dual1[i, :]  # 长度为 n_sv1
        coeffs2 = dual2[i, :]  # 长度为 n_sv2
        inner_i = np.dot(coeffs1, np.dot(K, coeffs2))
        inner_products.append(inner_i)
    
    inner_products = np.array(inner_products)
    
    if aggregation == 'average':
        return np.mean(inner_products)
    elif aggregation == 'sum':
        return np.sum(inner_products)
    else:
        raise ValueError("aggregation 必须为 'average' 或 'sum'")

def rkhs_norm_multiclass(model, kernel_func, gamma=0.1, aggregation='average'):
    """
    针对多分类 SVM 模型：
    对每个二分类器，计算自身在 RKHS 中的内积（即该二分类器的系数对自身计算内积），
    然后按指定方式汇总，再取平方根得到范数。
    """
    sv = model.support_vectors_
    K = kernel_func(sv, sv, gamma=gamma)
    
    dual = model.dual_coef_  # shape: (n_binary, n_sv)
    n_classifiers = dual.shape[0]
    inner_products = []
    for i in range(n_classifiers):
        coeffs = dual[i, :]
        inner_i = np.dot(coeffs, np.dot(K, coeffs))
        inner_products.append(inner_i)
    
    inner_products = np.array(inner_products)
    
    if aggregation == 'average':
        aggregated = np.mean(inner_products)
    elif aggregation == 'sum':
        aggregated = np.sum(inner_products)
    else:
        raise ValueError("aggregation 必须为 'average' 或 'sum'")
    
    return np.sqrt(aggregated)

def rkhs_cosine_similarity_multiclass(model1, model2, kernel_func, gamma=0.1, aggregation='average'):
    """
    针对多分类 SVM 模型：
    计算两个模型在 RKHS 空间中（基于各个二分类器）的余弦相似度。
    """
    ip = rkhs_inner_product_multiclass(model1, model2, kernel_func, gamma=gamma, aggregation=aggregation)
    norm1 = rkhs_norm_multiclass(model1, kernel_func, gamma=gamma, aggregation=aggregation)
    norm2 = rkhs_norm_multiclass(model2, kernel_func, gamma=gamma, aggregation=aggregation)
    return ip / (norm1 * norm2 + 1e-8)

def utility_RKHS(x_train, y_train, x_valid, y_valid, clf, final_model, gamma=0.1, kernel_func=rbf_kernel, aggregation='average'):
    """
    基于 Hilbert 空间（RKHS）的相似度效用函数。
    在子集 x_train, y_train 上训练 clf 得到子模型，
    返回子模型与 final_model 在 RKHS 中（针对每个二分类任务）的余弦相似度作为效用值。
    当抽到的子集只有一个类别时，返回 0。
    """
    # 如果子集只有一个类别，则返回 0
    if len(np.unique(y_train)) < 2:
        return 0.0
    else:
        try:
            clf.fit(x_train, y_train)
            similarity = rkhs_cosine_similarity_multiclass(clf, final_model, kernel_func, gamma=gamma, aggregation=aggregation)
        except Exception as e:
            print(f"模型训练失败，原因: {type(e).__name__}: {str(e)}")
            return 0.0
    return similarity

#####################################
# 2. 基于 KL 散度的预测相似性效用函数
#####################################

def utility_KL(x_train, y_train, x_valid, y_valid, clf, final_model, eps=1e-10):
    """
    基于预测概率的 KL 散度计算效用函数。
    在子集上训练 clf 后，在验证集上计算 clf 与 final_model 的预测概率，
    返回对称 KL 散度转换后的相似性（exp(-avg_sym_kl)）。
    当抽到的子集只有一个类别时，返回 0。
    """
    # 如果子集只有一个类别，则返回 0
    if len(np.unique(y_train)) < 2:
        return 0.0
    clf.fit(x_train, y_train)
    full_probs = final_model.predict_proba(x_valid)
    curr_probs = clf.predict_proba(x_valid)
    full_classes = final_model.classes_
    curr_classes = clf.classes_
    n_full = len(full_classes)
    mapped_curr_probs = np.zeros((curr_probs.shape[0], n_full))
    for i, cl in enumerate(curr_classes):
        indices = np.where(full_classes == cl)[0]
        if len(indices) > 0:
            mapped_curr_probs[:, indices[0]] = curr_probs[:, i]
    row_sums = mapped_curr_probs.sum(axis=1, keepdims=True)
    mapped_curr_probs = mapped_curr_probs / (row_sums + eps)
    kl_divs = []
    for fp, cp in zip(full_probs, mapped_curr_probs):
        fp = fp / (np.sum(fp) + eps)
        cp = cp / (np.sum(cp) + eps)
        kl1 = np.sum(fp * np.log((fp + eps) / (cp + eps)))
        kl2 = np.sum(cp * np.log((cp + eps) / (fp + eps)))
        kl_sym = 0.5 * (kl1 + kl2)
        kl_divs.append(kl_sym)
    avg_kl = np.mean(kl_divs)
    return np.exp(-avg_kl)

#####################################
# 3. 基于准确率的效用函数
#####################################

def utility_acc(x_train, y_train, x_valid, y_valid, clf, final_model):
    """
    基于准确率的效用函数。
    在子集上训练分类器，然后在验证集上计算准确率作为效用值。
    """
    single_pred_label = (True if len(np.unique(y_train)) == 1 else False)
    
    if single_pred_label:
        y_pred = [y_train[0]] * len(y_valid)
    else:
        try:
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_valid)
        except Exception as e:
            return 0
    
    return metrics.accuracy_score(y_valid, y_pred, normalize=True)

#####################################
# 4. 基于余弦相似度的效用函数
#####################################

def utility_cosine(x_train, y_train, x_valid, y_valid, clf, final_model):
    """
    基于模型参数余弦相似度的效用函数。
    适用于线性模型（如 LinearSVC, LogisticRegression）。
    """
    if len(np.unique(y_train)) < 2:
        return 0.0
    
    try:
        clf.fit(x_train, y_train)
        
        # 获取模型参数
        if hasattr(clf, 'coef_') and hasattr(final_model, 'coef_'):
            # 对于多分类情况，取平均余弦相似度
            if clf.coef_.ndim > 1:
                similarities = []
                for i in range(min(clf.coef_.shape[0], final_model.coef_.shape[0])):
                    coef1 = clf.coef_[i].flatten()
                    coef2 = final_model.coef_[i].flatten()
                    norm1 = np.linalg.norm(coef1)
                    norm2 = np.linalg.norm(coef2)
                    if norm1 > 1e-8 and norm2 > 1e-8:
                        sim = np.dot(coef1, coef2) / (norm1 * norm2)
                        similarities.append(sim)
                return np.mean(similarities) if similarities else 0.0
            else:
                coef1 = clf.coef_.flatten()
                coef2 = final_model.coef_.flatten()
                norm1 = np.linalg.norm(coef1)
                norm2 = np.linalg.norm(coef2)
                if norm1 > 1e-8 and norm2 > 1e-8:
                    return np.dot(coef1, coef2) / (norm1 * norm2)
                else:
                    return 0.0
        else:
            # 对于没有coef_属性的模型，退回到准确率
            return utility_acc(x_train, y_train, x_valid, y_valid, clf, final_model)
            
    except Exception as e:
        print(f"余弦相似度计算失败: {e}")
        return 0.0