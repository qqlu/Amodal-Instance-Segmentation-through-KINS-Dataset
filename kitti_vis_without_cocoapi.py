import numpy as np
import cv2
import cvbase as cvb
import os
import pycocotools.mask as maskUtils
import pdb

def make_json_dict(imgs, anns):
	imgs_dict = {}
	anns_dict = {}
	for ann in anns:
		image_id = ann["image_id"]
		if not image_id in anns_dict:
			anns_dict[image_id] = []
			anns_dict[image_id].append(ann)
		else:
			anns_dict[image_id].append(ann)
	
	for img in imgs:
		image_id = img['id']
		imgs_dict[image_id] = img['file_name']

	return imgs_dict, anns_dict

if __name__ == '__main__':
	src_img_path = "/home/qilu/Downloads/image/training/image_2"
	src_gt7_path = "/home/qilu/Amodal-Instance-Segmentation-through-KINS-Dataset/instances_train_1.json"
	anns = cvb.load(src_gt7_path)
	imgs_info = anns['images']
	anns_info = anns["annotations"]

	imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
	count = 0

	for img_id in anns_dict.keys():
		img_name = imgs_dict[img_id]

		img_path = os.path.join(src_img_path, img_name)
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		
		height, width, _ = img.shape
		anns = anns_dict[img_id]

		for ann in anns:
			amodal_rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
			amodal_ann_mask = maskUtils.decode(amodal_rle)
			inmodal_ann_mask = maskUtils.decode(ann['inmodal_seg'])[:,:,np.newaxis]

			amodal_ann_mask = amodal_ann_mask * 255
			inmodal_ann_mask = inmodal_ann_mask * 255
			# pdb.set_trace()
			amodal_ann_mask = np.concatenate((amodal_ann_mask, amodal_ann_mask, amodal_ann_mask), axis=2)
			inmodal_ann_mask = np.concatenate((inmodal_ann_mask, inmodal_ann_mask, inmodal_ann_mask), axis=2)
			
			show_img = np.concatenate((img, amodal_ann_mask, inmodal_ann_mask), axis=0)
			
			cv2.namedWindow("img-amodal-inmodal", cv2.WINDOW_NORMAL)
			cv2.imshow("img-amodal-inmodal", show_img)
			cv2.waitKey(0)


