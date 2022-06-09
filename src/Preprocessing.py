from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools import voxeldescriptors
from moleculekit.smallmol.smallmol import SmallMol
from htmd.ui import *
import numpy as np


class Preprocessing:
    def __init__(self, boxsize):
        self.boxsize = list(boxsize)

    def calcDatasetVoxel(self, protPath, ligPath):
        dataset = list()
        try:
            sm = SmallMol(ligPath, force_reading=True, fixHs=False)
            x = np.mean(sm.get('coords')[:, 0])
            y = np.mean(sm.get('coords')[:, 1])
            z = np.mean(sm.get('coords')[:, 2])
            fs, cs, ns = voxeldescriptors.getVoxelDescriptors(
                sm,
                center=[x, y, z],
                boxsize=self.boxsize,
                version=2
            )
        except:
            print('Ligand ' + ligPath + ' leads to errors!')
            return 0
        try:
            f, c, n = self.calcProtVoxel(x, y, z, protPath)
            feature_protein = f
            feature_protein_shaped = f.reshape(n[0], n[1], n[2], f.shape[1])
            feature_ligand = fs
            feature_ligand_shaped = fs.reshape(ns[0], ns[1], ns[2], fs.shape[1])
            datapoint = np.concatenate((feature_protein_shaped, feature_ligand_shaped), axis=3).transpose([3, 0, 1, 2])
            dataset.append(datapoint)
        except:
            return 0

        return np.array(dataset), np.array(c), np.array(feature_protein), np.array(feature_ligand), np.array(feature_protein_shaped), np.array(feature_ligand_shaped)

    def calcProtVoxel(self, x, y, z, protPath):
        try:
            prot = Molecule(protPath)
            if prot.numAtoms > 50000:
                factorx = self.boxsize[0] * 2.5
                factory = self.boxsize[1] * 2.5
                factorz = self.boxsize[2] * 2.5
                prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
            prot.filter('protein')
            prot.bonds = prot._getBonds()
            prot = prepareProteinForAtomtyping(prot)
            prot.set(value='Se', field='element', sel='name SE')

            # from moleculekit.tools.atomtyper import (
            #     getFeatures,
            #     getPDBQTAtomTypesAndCharges,
            # )
            #
            # prot.atomtype, prot.charge = getPDBQTAtomTypesAndCharges(prot)

            f, c, n = voxeldescriptors.getVoxelDescriptors(
                prot,
                center=[x, y, z],
                boxsize=self.boxsize,
                version=2
            )

        except:
            try:
                prot = Molecule(protPath)
                if prot.numAtoms > 50000:
                    factorx = self.boxsize[0] * 2.5
                    factory = self.boxsize[1] * 2.5
                    factorz = self.boxsize[2] * 2.5
                    prot.filter('z < ' + format(z + factorz) + ' and z > ' + format(z - factorz))
                    prot.filter('x < ' + format(x + factorx) + ' and x > ' + format(x - factorx))
                    prot.filter('y < ' + format(y + factory) + ' and y > ' + format(y - factory))
                prot.filter('protein')
                prot.filter('not resname 3EB')
                prot = proteinPrepare(prot)
                prot = autoSegment(prot)
                prot.set(value='Se', field='element', sel='name SE')
                try:
                    prot.mutateResidue('resname TPO', 'THR')
                except:
                    pass
                try:
                    prot.mutateResidue('resname MSE', 'MET')
                except:
                    pass
                try:
                    prot.mutateResidue('resname SEP', 'SER')
                except:
                    pass
                prot = charmm.build(prot, ionize=False)

                # from moleculekit.tools.atomtyper import (
                #     getFeatures,
                #     getPDBQTAtomTypesAndCharges,
                # )
                #
                # prot.atomtype, prot.charge = getPDBQTAtomTypesAndCharges(prot)

                f, c, n = voxeldescriptors.getVoxelDescriptors(
                    prot,
                    center=[x, y, z],
                    boxsize=self.boxsize,
                    version=2
                )
            except:
                print('Protein ' + protPath + ' leads to errors!')
                return 0
        return f, c, n
