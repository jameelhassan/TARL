import warnings
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tarl.datasets.dataloader.SemanticKITTITemporal import TemporalKITTISet
from tarl.datasets.dataloader.DINOSemanticKITTI import DINOKITTISet
from tarl.datasets.dataloader.SemanticKITTI import KITTISet
from tarl.datasets.dataloader.DataloaderTemplate import TemplateSet
from tarl.utils.collations import SparseAugmentedCollation
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings('ignore')

__all__ = ['DINOSemKITTIDDP', 'DINOSemKITTI', 'TemporalSemKITTIDDP', 'TemporalKittiDataModule', 'KittiDataModule', 'TemplateDataModule']

class DINOKittiDDPDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        augmented_dir = self.cfg['data'].get('augmented_dir', 'NewOptPCD_T12')
        data_set = DINOKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            scan_window=self.cfg['train']['scan_window'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            use_ground_pred=self.cfg['data']['use_ground_pred'],
            num_points=self.cfg['train']['num_points'],
            augmented_dir=augmented_dir,
            teacher_drop_rate=self.cfg['data']['teacher_drop_rate'],
            student_drop_rate=self.cfg['data']['student_drop_rate']
            )
    
        self.train_sampler = DistributedSampler(data_set)
        batch_size_per_gpu = self.cfg['train'].get('batch_size') // self.cfg['train'].get('n_gpus')
        loader = DataLoader(
            data_set, 
            batch_size=batch_size_per_gpu,  # Ensure batch size is set in your config
            num_workers=self.cfg['train']['num_workers'],
            collate_fn=collate,
            sampler=self.train_sampler,
            pin_memory=True,
        )
        return loader
    
class DINOKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        data_set = DINOKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            scan_window=self.cfg['train']['scan_window'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            use_ground_pred=self.cfg['data']['use_ground_pred'],
            num_points=self.cfg['train']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader
    
class TemporalKittiDDPDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        augmented_dir = self.cfg['data'].get('augmented_dir', 'segments_views')
        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            scan_window=self.cfg['train']['scan_window'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            use_ground_pred=self.cfg['data']['use_ground_pred'],
            num_points=self.cfg['train']['num_points'],
            augmented_dir=augmented_dir
            )
    
        self.train_sampler = DistributedSampler(data_set)
        batch_size_per_gpu = self.cfg['train'].get('batch_size') // self.cfg['train'].get('n_gpus')
        loader = DataLoader(
            data_set, 
            batch_size=batch_size_per_gpu,  # Ensure batch size is set in your config
            num_workers=self.cfg['train']['num_workers'],
            collate_fn=collate,
            sampler=self.train_sampler,
            pin_memory=True,
        )
        return loader
    
class TemporalKittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        data_set = TemporalKITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            scan_window=self.cfg['train']['scan_window'],
            split=self.cfg['data']['split'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            use_ground_pred=self.cfg['data']['use_ground_pred'],
            num_points=self.cfg['train']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class KittiDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        data_set = KITTISet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            pre_training=self.cfg['train']['pre_train'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            num_points=self.cfg['train']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class TemplateDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        # this collation list points and objects views and create a dict out of it  
        collate = SparseAugmentedCollation(self.cfg['train']['resolution'])

        data_set = TemplateSet(
            data_dir=self.cfg['data']['data_dir'],
            seqs=self.cfg['data']['train'],
            split=self.cfg['data']['split'],
            pre_training=self.cfg['train']['pre_train'],
            resolution=self.cfg['train']['resolution'],
            percentage=self.cfg['data']['percentage'],
            intensity_channel=self.cfg['data']['intensity'],
            num_points=self.cfg['train']['num_points'])
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

data_modules = {
    'DINOSemKITTIDDP': DINOKittiDDPDataModule,
    'DINOSemKITTI': DINOKittiDataModule,
    'TemporalSemKITTIDDP': TemporalKittiDDPDataModule,
    'TemporalSemKITTI': TemporalKittiDataModule,
    'SemKITTI': KittiDataModule,
    'TemplateDataset': TemplateDataModule,
}
