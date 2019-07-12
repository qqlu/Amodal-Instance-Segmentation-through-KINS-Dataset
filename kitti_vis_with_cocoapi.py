import os
from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import pylab
import skimage.io as io

if __name__ == '__main__':
	src_img_path = "/home/qilu/Downloads/image/training/image_2"
	src_gt7_path = "/home/qilu/Amodal-Instance-Segmentation-through-KINS-Dataset/instances_train_1.json"

	coco = COCO(src_gt7_path)
	imgIds = coco.getImgIds()
	# anns_dict_17 = make_json_dict(anns_2017)
	# pdb.set_trace()
	count = 0

	for img_id in imgIds:
		
		# pdb.set_trace()
		img= coco.loadImgs(img_id)[0]
		img_name = img['file_name']

		img_path = os.path.join(src_img_path, img_name)
		I = io.imread(img_path)
		plt.imshow(I)
		plt.axis('off')
		annIds = coco.getAnnIds(imgIds=img_id)
		anns = coco.loadAnns(annIds)
		coco.showAnns(anns)
		plt.show()


