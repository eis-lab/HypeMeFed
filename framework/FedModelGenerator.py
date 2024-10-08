class FedModelGenerator:
    FILENAME_CLASSES = {
        'unimib': {
            'vgg': ('vgg', 'vgg_bn'),
            'multiexit_vgg': ('vgg', 'multiexit_vgg_bn'),
        },
        'stl10': {
            'multiexit_vgg': ('vgg', 'multiexit_vgg_bn'),
        },
        'svhn': {
            'vgg': ('vgg', 'vgg_bn'),
            'multiexit_vgg': ('vgg', 'multiexit_vgg_bn'),
        }
    }

    @staticmethod
    def generate_from_name_for_dataset(dataset_name: str, model_name: str, model_params:dict={}):
        dataset_name = dataset_name.lower()
        model_name = model_name.lower()

        if dataset_name not in FedModelGenerator.FILENAME_CLASSES:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        dataset_models = FedModelGenerator.FILENAME_CLASSES[dataset_name]

        if model_name not in dataset_models:
            raise ValueError(f"Unknown model name: {model_name} for dataset {dataset_name}")

        filename, model_class_name = dataset_models[model_name]

        module_name = f"models.{dataset_name}.{filename}"
        module = __import__(module_name, fromlist=[filename])
        model_class = getattr(module, model_class_name)

        if model_params is None:
            model_params = {}
        return model_class(**model_params)
