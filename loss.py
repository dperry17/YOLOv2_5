import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, num_classes, num_anchors, lambda_coord=5, lambda_noobj=0.5):
        super(Loss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')

    def forward(self, predictions, targets, enchors):

        batch, grid, _, _, _ = predictions.shape

        pred_tx = predictions[..., 0]
        pred_ty = predictions[..., 1]
        pred_tw = predictions[..., 2]
        pred_th = predictions[..., 3]
        pred_obj = predictions[..., 4]
        pred_class = predictions[..., 5]

        true_tx = targets[..., 0]
        true_ty = targets[..., 1]
        true_tw = targets[..., 2]
        true_th = targets[..., 3]
        true_obj = targets[..., 4]
        true_class = targets[..., 5]

        loss_coord = self.mse(pred_tx, true_tx) + self.mse(pred_tx, true_tx) + self.mse(pred_th, true_th) + self.mse(pred_tw, true_tw)
        loss_coord = loss_coord * lambda_coord

        loss_obj = self.bce(pred_obj, true_obj)
        loss_class = self.bce(pred_class, true_class)

        no_obj_mask = (true_obj == 0)
        loss_noobj= self.bce(pred_obj[no_obj_mask], true_obj[no_obj_mask])
        loss_noobj = loss_noobj * lambda_noobj

        return loss_coord + loss_obj + loss_class + loss_noobj
