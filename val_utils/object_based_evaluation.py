from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage as ndi

from utils.utils import compute_divergence


def match_objects(pred_props, target_props, max_distance=50):
    """
    使用匈牙利算法匹配预测对象与目标对象

    Args:
        pred_props: 预测对象属性列表
        target_props: 目标对象属性列表
        max_distance: 最大匹配距离(网格单位)

    Returns:
        matches: 匹配对列表，每个元素包含 pred_id, target_id, displacement
    """
    if len(pred_props) == 0 or len(target_props) == 0:
        return []

    pred_centers = np.array([[p['centroid_y'], p['centroid_x']] for p in pred_props])
    target_centers = np.array([[t['centroid_y'], t['centroid_x']] for t in target_props])

    cost_matrix = cdist(pred_centers, target_centers)

    cost_matrix[cost_matrix > max_distance] = max_distance + 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for pred_idx, target_idx in zip(row_ind, col_ind):
        if cost_matrix[pred_idx, target_idx] <= max_distance:
            displacement = cost_matrix[pred_idx, target_idx]
            matches.append({
                'pred_id': pred_idx,
                'target_id': target_idx,
                'displacement': displacement
            })

    return matches


def compute_match_metrics(matches):
    """
    计算匹配指标统计

    Args:
        matches: 匹配对列表

    Returns:
        metrics: 包含各指标均值的字典
    """
    if len(matches) == 0:
        return {
            'mean_displacement': np.nan,
            'median_displacement': np.nan,
            'match_rate': 0.0
        }

    displacements = [m['displacement'] for m in matches]

    return {
        'mean_displacement': np.mean(displacements),
        'median_displacement': np.median(displacements),
        'std_displacement': np.std(displacements),
        'match_count': len(matches)
    }


def object_based_evaluation(pred, target, u, v, min_distance=5, threshold=1.0):
    """
    基于对象的评估：识别降水单体，匹配位置

    参考：MODE (Method for Object-based Diagnostic Evaluation)
    """
    # 识别降水对象（连通域）
    pred_objects = identify_objects(pred, threshold, min_distance)
    target_objects = identify_objects(target, threshold, min_distance)

    # 计算每个对象的属性
    pred_props = compute_object_properties(pred, pred_objects)
    target_props = compute_object_properties(target, target_objects)

    # 对象匹配（匈牙利算法）
    matches = match_objects(pred_props, target_props)

    # 按风场条件分组评估
    results = {
        'all': compute_match_metrics(matches),
        'strong_wind': [],
        'convergence': []
    }

    for match in matches:
        pred_obj = pred_props[match['pred_id']]
        target_obj = target_props[match['target_id']]

        # 获取对象中心的风场条件
        cy, cx = int(target_obj['centroid_y']), int(target_obj['centroid_x'])
        wind_speed = np.sqrt(u[cy, cx] ** 2 + v[cy, cx] ** 2)
        div = compute_divergence(u, v)[cy, cx]

        # 分类
        obj_result = {
            'displacement': match['displacement'],  # 位移误差(km)
            'intensity_ratio': pred_obj['max_intensity'] / target_obj['max_intensity'],
            'area_ratio': pred_obj['area'] / target_obj['area'],
            'wind_speed': wind_speed,
            'divergence': div
        }

        if wind_speed > np.percentile(u ** 2 + v ** 2, 75):
            results['strong_wind'].append(obj_result)
        if div < np.percentile(compute_divergence(u, v), 25):
            results['convergence'].append(obj_result)

    return results


def identify_objects(field, threshold, min_distance):
    """识别降水对象"""
    # 二值化
    binary = field > threshold

    # 找到局部极大值作为种子
    peaks = peak_local_max(field, min_distance=min_distance,
                           threshold_abs=threshold)

    # 分水岭或区域生长
    markers = np.zeros_like(field, dtype=int)
    for i, (y, x) in enumerate(peaks):
        markers[y, x] = i + 1

    # 使用距离变换+分水岭
    distance = ndi.distance_transform_edt(binary)
    objects = watershed(-distance, markers, mask=binary)

    return objects


def compute_object_properties(field, objects):
    """计算对象属性"""
    props = []
    for obj_id in range(1, objects.max() + 1):
        mask = objects == obj_id
        if mask.sum() == 0:
            continue

        props.append({
            'id': obj_id,
            'area': mask.sum(),
            'centroid_y': np.mean(np.where(mask)[0]),
            'centroid_x': np.mean(np.where(mask)[1]),
            'max_intensity': field[mask].max(),
            'mean_intensity': field[mask].mean(),
            'bbox': (np.min(np.where(mask)[0]), np.max(np.where(mask)[0]),
                     np.min(np.where(mask)[1]), np.max(np.where(mask)[1]))
        })

    return props 


if __name__ == '__main__':
    
    results = object_based_evaluation(
        pred=predicted_precip,   # 预测降水
        target=observed_precip,  # 观测降水
        u=wind_u,                # 风场U分量
        v=wind_v,                # 风场V分量
        threshold=1.0
    )

    # 强风区精度指标
    print("强风区位移误差:", 
        np.mean([r['displacement'] for r in results['strong_wind']]))