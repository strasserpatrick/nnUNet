import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.pcrlv2_dataset import PCRLv2Dataset


class nnUNetDataLoaderPCRLv2(nnUNetDataLoaderBase):

    def determine_shapes(self):
        return 0, 0

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = []
        case_properties = None

        for j, i in enumerate(selected_keys):
            data, seg, properties = self._data.load_case(i)
            data_all.append(data)

            # properties is same for every case in this implementation
            if case_properties is None:
                case_properties = properties
            
        return {'data': data_all, 'seg': None, 'properties': case_properties, 'keys': selected_keys}


"""
TODO: I think that the data augmentations need to go here:,
but we have to make this debuggable some how

    def __getitem__(self, index):
        image_name = self.imgs[index]
        pair = np.load(image_name)
        crop1 = pair[0]
        crop1 = np.expand_dims(crop1, axis=0)
        crop2 = pair[1]
        crop2 = np.expand_dims(crop2, axis=0)
        # gt1 = torch.tensor(gt1, dtype=torch.float)
        # gt2 = torch.tensor(gt2, dtype=torch.float)
        input1 = self.transform(crop1)
        input2 = self.transform(crop2)
        gt1 = copy.deepcopy(input1)
        gt2 = copy.deepcopy(input2)
        input1 =self.global_transforms(input1)
        input2 = self.global_transforms(input2)
        # input1 = self.local_pixel_shuffling(input1, prob=self.config.local_rate)
        # input2 = self.local_pixel_shuffling(input2, prob=self.config.local_rate)
        # if random.random() < self.config.paint_rate:
        #     input1 = self.image_in_painting(input1)
        #     input2 = self.image_in_painting(input2)
            # if random.random() < self.config.inpaint_rate:
            #     # Inpainting
            #     input1 = self.image_in_painting(input1)
            #     input2 = self.image_in_painting(input2)
            # else:
            #     # Outpainting
            #     input1 = self.image_out_painting(input1)
            #     input2 = self.image_out_painting(input2)
        locals = np.load(image_name.replace('global', 'local'))
        local_inputs = []
        # local_inputs = []
        for i  in range(locals.shape[0]):
            img = locals[i]
            img = np.expand_dims(img, axis=0)
            img = self.transform(img)
            img = self.local_transforms(img)
            # img = self.local_pixel_shuffling(img, prob=self.config.local_rate, num_block=1000)
            local_inputs.append(img)
        # for local_path in local_paths:
        #     img = np.load(local_path)
        #     img = np.expand_dims(img, axis=0)
        #     img = self.transform(img)
        #     # img = self.local_pixel_shuffling(img, prob=self.config.local_rate, num_block=1000)
        #     # if random.random() < self.config.paint_rate - 0.5:
        #     #     if random.random() < self.config.inpaint_rate:
        #     #         # Inpainting
        #     #         img = self.image_in_painting(img, cnt=3)
        #     #     else:
        #     #         # Outpainting
        #     #         img = self.image_out_painting(img, cnt=2)
        #     local_inputs.append(torch.tensor(img, dtype=torch.float))
        return torch.tensor(input1, dtype=torch.float), torch.tensor(input2, dtype=torch.float), \
               torch.tensor(gt1, dtype=torch.float), \
               torch.tensor(gt2, dtype=torch.float), local_input
"""

