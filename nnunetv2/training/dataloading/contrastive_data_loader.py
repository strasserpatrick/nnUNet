from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase


class nnUNetContrastiveDataLoader(nnUNetDataLoaderBase):

    def determine_shapes(self):
        return 0, 0

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = []
        case_properties = []

        for j, i in enumerate(selected_keys):
            data, seg, properties = self._data.load_case(i)
            data_all.append(data)
            case_properties.append(properties)

        return {'data': data_all, 'seg': None, 'properties': case_properties, 'keys': selected_keys}
