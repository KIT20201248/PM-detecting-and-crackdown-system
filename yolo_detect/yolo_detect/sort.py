# SORT: A Simple, Online and Realtime Tracker
# ─────────────────────────────────────────────
# 칼만필터와 IOU를 이용한 객체 추적 코드
# - 입력: [x1,y1,x2,y2,score]
# - 출력: [x1,y1,x2,y2,id]

import numpy as np
from filterpy.kalman import KalmanFilter

np.random.seed(0)


# ── [1] 선형 할당 (Hungarian algorithm) ───────────────────────────────
def linear_assignment(cost_matrix):
  # Hungarian 알고리즘으로 매칭 계산
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


# ── [2] IOU 계산 ──────────────────────────────────────────────────────
def iou_batch(bb_test, bb_gt):
  # IOU(교집합/합집합 비율) 계산
  # 입력: [x1,y1,x2,y2] 형식
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  # 겹치는 영역 좌표 계산
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  # IOU = 교집합 / (합집합)
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
             (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
  return o


# ── [3] BBox 변환 함수 ────────────────────────────────────────────────
def convert_bbox_to_z(bbox):
  # [x1,y1,x2,y2] → [cx,cy,area,ratio]
  # - cx, cy: 중심 좌표
  # - area: 박스 넓이
  # - ratio: 가로세로 비율
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
  # [cx,cy,area,ratio] → [x1,y1,x2,y2] 변환
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if score is None:
    return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
  else:
    return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# ── [4] KalmanBoxTracker: 개별 객체 추적기 ─────────────────────────────
class KalmanBoxTracker(object):
  # 하나의 객체에 대해 Kalman Filter로 bbox 추적
  count = 0

  def __init__(self, bbox):
    # bbox 초기값으로 tracker 생성
    self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 7차원 상태, 4차원 관측
    # 상태 전이 행렬(F) → 위치 + 속도 모델
    self.kf.F = np.array([[1,0,0,0,1,0,0],
                          [0,1,0,0,0,1,0],
                          [0,0,1,0,0,0,1],
                          [0,0,0,1,0,0,0],
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])
    # 관측 행렬(H)
    self.kf.H = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]])

    # 잡음/불확실성 초기화
    self.kf.R[2:, 2:] *= 10.
    self.kf.P[4:, 4:] *= 1000.
    self.kf.P *= 10.
    self.kf.Q[-1, -1] *= 0.01
    self.kf.Q[4:, 4:] *= 0.01

    # 초기 상태를 bbox로 설정
    self.kf.x[:4] = convert_bbox_to_z(bbox)

    # 내부 상태 변수
    self.time_since_update = 0  # 마지막 업데이트 이후 frame 수
    self.id = KalmanBoxTracker.count  # 고유 ID
    KalmanBoxTracker.count += 1
    self.history = []           # 예측 히스토리
    self.hits = 0               # 총 업데이트 횟수
    self.hit_streak = 0         # 연속 업데이트 성공 횟수
    self.age = 0                # tracker 수명

  def update(self, bbox):
    # 새 detection으로 상태 갱신
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    # 다음 위치 예측
    if (self.kf.x[6] + self.kf.x[2]) <= 0:
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if self.time_since_update > 0:
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    # 현재 bbox 반환
    return convert_x_to_bbox(self.kf.x)


# ── [5] detection ↔ tracker 매칭 ──────────────────────────────────────
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
  # detection과 tracker를 IOU로 매칭
  # 반환: matches, unmatched_detections, unmatched_trackers
  if len(trackers) == 0:
    return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)  # Hungarian 알고리즘
  else:
    matched_indices = np.empty(shape=(0, 2))

  # 매칭 실패한 detection/tracker 분류
  unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
  unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

  # IOU 낮은 매칭 제거
  matches = []
  for m in matched_indices:
    if iou_matrix[m[0], m[1]] < iou_threshold:
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1, 2))

  if len(matches) == 0:
    matches = np.empty((0, 2), dtype=int)
  else:
    matches = np.concatenate(matches, axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ── [6] SORT 클래스 (메인 추적기) ──────────────────────────────────────
class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    # max_age: 감지 안 돼도 tracker를 유지할 frame 수
    # min_hits: tracker가 '정식 ID' 받기 전 최소 감지 수
    # iou_threshold: detection-tracker 매칭 기준 IOU
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    # 한 프레임에 대해 추적 업데이트
    # 입력: dets = [x1,y1,x2,y2,score]
    # 출력: [x1,y1,x2,y2,id]
    self.frame_count += 1
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []

    # (1) 기존 tracker로 예측
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)

    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    # (2) detection과 tracker 매칭
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # (3) 매칭된 tracker 업데이트
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # (4) 매칭 안 된 detection → 새 tracker 생성
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i, :])
      self.trackers.append(trk)

    # (5) 유효한 tracker만 결과로 반환
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state()[0]
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        # tracker.id + 1 : MOT 챌린지 형식에 맞추기 위해 +1
        ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
      i -= 1
      if trk.time_since_update > self.max_age:
        # 오래 감지 안 된 tracker는 제거
        self.trackers.pop(i)

    if len(ret) > 0:
      return np.concatenate(ret)
    return np.empty((0, 5))
