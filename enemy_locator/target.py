from enemy_locator.tracker import TrackerPool

classes = ["car", "armor1red", "armor2red", "armor3red", "armor4red", "armor5red",
           "armor1blue", "armor2blue", "armor3blue", "armor4blue", "armor5blue", "base", "ignore"]

cls2id = [-1,
          1, 2, 3, 4, 5,
          101, 102, 103, 104, 105]

red_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
blue_id = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}


class Car:
    def __init__(self, bbox, cls=-1, conf=0,):
        self.cls = cls
        self.car_id = -1
        self.color = -1
        self.conf = conf

        self.bbox = bbox
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.rect = [self.x1, self.y1, self.w, self.h]
        self.center_yx = ((self.y1 + self.y2) / 2, (self.x1 + self.x2) / 2)
        self.x = self.center_yx[1]
        self.y = self.center_yx[0]

        self.life_span = 5
        self.armors = []

    def get_id(self):
        return cls2id[self.cls]

    def get_idx_in_targets(self, color):
        return self.car_id - 100 * color - 1

    def refresh(self):
        if self.life_span <= 0 < self.conf:
            self.conf = 0
            self.armors.clear()
        else:
            self.life_span -= 1

    def match(self, armor):
        if self.y1 < armor.center_xy[1] < self.y2 and \
                self.x1 < armor.center_xy[0] < self.x2:
            self.armors.append(armor)

            if len(self.armors) > 1:
                self.cls = self.armors[0].conf > self.armors[1].conf \
                           and self.armors[0].cls \
                           or self.armors[1].cls
                self.conf = self.armors[0].conf > self.armors[1].conf \
                            and self.armors[0].conf \
                            or self.armors[1].conf
            else:
                self.cls = armor.cls
                self.conf = armor.conf

            self.color = self.armors[0].color
            self.car_id = self.get_id()

    def copy(self):
        car = Car(bbox=self.bbox, cls=self.cls, conf=self.conf)
        for armor in self.armors:
            car.armors.append(armor.copy())
        return car

    def update(self, cls, conf):
        self.cls = cls
        self.conf = conf
        self.life_span = 5


class Armor:
    def __init__(self, bbox, color=-1, cls=-1, conf=0,):
        self.color = color
        self.cls = cls
        self.conf = conf

        self.bbox = bbox
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1
        self.rect = [self.x1, self.y1, self.w, self.h]
        self.center_xy = ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
        self.x = self.center_xy[0]
        self.y = self.center_xy[1]
        # self.life_span = 5

    def get_id(self):
        return cls2id[self.cls]

    def copy(self):
        return Armor(bbox=self.bbox, color=self.color, cls=self.cls, conf=self.conf)


class Targets:
    def __init__(self, color, thresh=0.7):
        self.enemy_color = color
        self.targets = [Car(bbox=[-1, -1, -1, -1], cls=_ + 100 * color) for _ in range(5)]
        # self.targets = [None for _ in range(5)]

        self.multi_tracker = TrackerPool(lost_thresh=10)
        self.iou_thresh = thresh

    def get_cars(self, boxes):
        cars = []
        armors = []
        for box in boxes:
            if int(box[5]) == 0:
                cars.append(Car(bbox=[box[0], box[1], box[2], box[3]],
                                cls=-1,
                                conf=box[4]))
            elif self.enemy_color == 0 and 1 <= int(box[5]) <= 5:  # enemy color
                armors.append(Armor(bbox=[box[0], box[1], box[2], box[3]],
                                    color=self.enemy_color,
                                    cls=int(box[5]),
                                    conf=box[4]
                                    )
                              )
            elif self.enemy_color == 1 and 6 <= int(box[5]) <= 10:
                armors.append(Armor(bbox=[box[0], box[1], box[2], box[3]],
                                    color=self.enemy_color,
                                    cls=int(box[5]),
                                    conf=box[4]
                                    )
                              )
        cars.sort(key=lambda x: x.conf, reverse=True)

        for car in cars:
            for armor in armors:
                car.match(armor)
        return cars

    def update_with_trackerpool(self, boxes, frame):
        cars = self.get_cars(boxes)

        pred_rects = []
        pred_bboxes = []
        bbox_owners = []
        owner2i = [-1 for _ in range(5)]
        for i in range(5):
            if self.multi_tracker.is_tracked[i]:
                pred_rect = self.multi_tracker.get_pred_rect_for(i, frame)
                if pred_rect is not None:
                    pred_rects.append(pred_rect)
                    owner2i[i] = len(bbox_owners)
                    bbox_owners.append(i)

        for i, rect in enumerate(pred_rects):
            pred_bboxes.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])

        self.multi_tracker.display_part(frame, pred_rects, bbox_owners)

        current_targets = [Car(bbox=[-1, -1, -1, -1], conf=0) for _ in range(5)]
        updated = [False for _ in range(5)]
        for car in cars:
            if len(car.armors) == 0:
                car.conf = 0  # IMPORTANT
                for idx, bbox in enumerate(pred_bboxes):
                    if iou(car.bbox, bbox) >= self.iou_thresh:
                        owner_id = bbox_owners[idx]
                        cls = owner_id + 5 * self.enemy_color + 1
                        car.update(cls=cls, conf=self.multi_tracker.tracker_conf[owner_id])
                        current_targets[owner_id] = car
            if car.conf > 0:
                idx = car.get_idx_in_targets(self.enemy_color)
                if car.conf > current_targets[idx].conf:
                    current_targets[idx] = car
                    updated[idx] = True

        # arrange trackers
        for i in range(5):
            if self.targets[i].conf < current_targets[i].conf:
                if self.multi_tracker.is_tracked[i]:
                    # idx = self.tracker_target.index(i)
                    if iou(current_targets[i].bbox, pred_bboxes[owner2i[i]]) >= self.iou_thresh:
                        pass
                    else:
                        self.multi_tracker.remove_tracker_for(i)
                        self.multi_tracker.set_tracker_for(i, frame, current_targets[i].rect)
                else:
                    self.multi_tracker.init_tracker_for(i, frame, current_targets[i].rect)

                self.targets[i] = current_targets[i]

        for i in range(5):
            if not updated[i]:
                self.targets[i].refresh()

    def update(self, boxes):
        cars = self.get_cars(boxes)

        current_targets = [Car(bbox=[-1, -1, -1, -1], conf=0) for _ in range(5)]
        updated = [False for _ in range(5)]
        for car in cars:
            if len(car.armors) == 0:
                car.conf = 0
            if car.conf > 0:
                idx = car.get_idx_in_targets(self.enemy_color)
                if car.conf > current_targets[idx].conf:
                    current_targets[idx] = car

        # arrange trackers
        for i in range(5):
            if self.targets[i].conf < current_targets[i].conf:
                updated[i] = True
                self.targets[i] = current_targets[i]

        for i in range(5):
            if not updated[i]:
                self.targets[i].refresh()


def iou(box1, box2):
    xmin_1, ymin_1, xmax_1, ymax_1 = box1
    xmin_2, ymin_2, xmax_2, ymax_2 = box2

    x_inter1 = max(xmin_1, xmin_2)
    y_inter1 = max(ymin_1, ymin_2)
    x_inter2 = min(xmax_1, xmax_2)
    y_inter2 = min(ymax_1, ymax_2)

    # 计算交集部分面积，因为图像是像素点，所以计算图像的长度需要加一
    # 比如有两个像素点(0,0)、(1,0)，那么图像的长度是1-0+1=2，而不是1-0=1
    inter_area = max(0, x_inter2 - x_inter1 + 1) * max(0, y_inter2 - y_inter1 + 1)
    box1_area = (xmax_1 - xmin_1 + 1) * (ymax_1 - ymin_1 + 1)
    box2_area = (xmax_2 - xmin_2 + 1) * (ymax_2 - ymin_2 + 1)

    _iou = inter_area / (box1_area + box2_area - inter_area)
    return _iou
