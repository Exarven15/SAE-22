import soundfile as sf

import matplotlib.pyplot as plt
import numpy as np
from sounddevice import *
from pylab import *
import scipy.signal as sc 
from pydub import *
# function by Ruellet & Fouquet

def aff_flac(fichier):
    # Charger le fichier audio FLAC
    audio, frequence_echantillonnage = sf.read(fichier)
    # Obtenir les informations sur le fichier audio
    nombre_pistes = audio.shape[1] if len(audio.shape) > 1 else 1
    info_fichier = sf.info(fichier)
    nombre_bits_par_echantillon = info_fichier.bits if hasattr(info_fichier, "bits") else None
    nombre_echantillons = audio.shape[0]
    duree_totale = audio.shape[0] / frequence_echantillonnage
    # Afficher les caractéristiques
    print("Caractéristiques du fichier audio:", fichier)
    print("Fréquence d'échantillonnage :", frequence_echantillonnage, "Hz")
    print("Nombre de pistes :", nombre_pistes)
    if nombre_bits_par_echantillon is not None:
        print("Nombre de bits par échantillon :", nombre_bits_par_echantillon)
    print("Nombre d'échantillons :", nombre_echantillons)
    print("Durée totale :", duree_totale, "secondes")

def obtenir_caracteristiques_audio(fichier):
    # Charger le fichier audio FLAC
    audio, frequence_echantillonnage = sf.read(fichier)
    # Obtenir les informations sur le fichier audio
    nombre_pistes = audio.shape[1] if len(audio.shape) > 1 else 1
    info_fichier = sf.info(fichier)
    nombre_bits_par_echantillon = info_fichier.bits if hasattr(info_fichier, "bits") else None
    nombre_echantillons = audio.shape[0]
    duree_totale = audio.shape[0] / frequence_echantillonnage
    # Créer un dictionnaire avec les caractéristiques
    caracteristiques = {
        "fichier": fichier,
        "frequence_echantillonnage": frequence_echantillonnage,
        "nombre_pistes": nombre_pistes,
        "nombre_bits_par_echantillon": nombre_bits_par_echantillon,
        "nombre_echantillons": nombre_echantillons,
        "duree_totale": duree_totale
    }
    return caracteristiques

def lire_flac(fichier):
    # Charger le fichier audio FLAC
    audio, frequence_echantillonnage = sf.read(fichier)
    # Renvoyer les échantillons de chaque piste et la fréquence d'échantillonnage
    return audio, frequence_echantillonnage

def afficher_pistes(extrait_audio, frequence_echantillonnage):
    # Obtenir les informations sur le nombre de pistes
    nombre_pistes = extrait_audio.shape[1] if len(extrait_audio.shape) > 1 else 1
    # Créer une grille de sous-graphes avec 2 lignes et 2 colonnes
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # Piste 1
    piste1_audio = extrait_audio[:, 0] if nombre_pistes > 1 else extrait_audio
    axs[0, 0].plot(np.arange(len(piste1_audio)) / frequence_echantillonnage, piste1_audio, color='purple')
    axs[0, 0].set_title("Piste 1 - Représentation temporelle")
    axs[0, 0].set_xlabel("Temps (s)")
    axs[0, 0].set_ylabel("Amplitude")
    amplitudes_piste1 = np.abs(np.fft.fft(piste1_audio))
    frequences_piste1 = np.fft.fftfreq(len(piste1_audio), 1 / frequence_echantillonnage) 
    axs[0, 1].plot(frequences_piste1, amplitudes_piste1, color='purple')
    axs[0, 1].set_title("Piste 1 - Spectre d'amplitude")
    axs[0, 1].set_xlabel("Fréquence (Hz)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].set_xlim(-500, 500) # la fréquence min et max 
    axs[0, 1].set_ylim(0, np.max(amplitudes_piste1) + 1000) # amplitude min et max 
    # Piste 2
    if nombre_pistes > 1:
        piste2_audio = extrait_audio[:, 1]
        axs[1, 0].plot(np.arange(len(piste2_audio)) / frequence_echantillonnage, piste2_audio, color='purple')
        axs[1, 0].set_title("Piste 2 - Représentation temporelle")
        axs[1, 0].set_xlabel("Temps (s)")
        axs[1, 0].set_ylabel("Amplitude")
        
        amplitudes_piste2 = np.abs(np.fft.fft(piste2_audio))
        frequences_piste2 = np.fft.fftfreq(len(piste2_audio), 1 / frequence_echantillonnage) 
        axs[1, 1].plot(frequences_piste2, amplitudes_piste2, color='purple')
        axs[1, 1].set_title("Piste 2 - Spectre d'amplitude")
        axs[1, 1].set_xlabel("Fréquence (Hz)")
        axs[1, 1].set_ylabel("Amplitude")
        axs[1, 1].set_xlim(-500, 500) # la fréquence min et max 
        axs[1, 1].set_ylim(0, np.max(amplitudes_piste2) + 1000) # amplitude min et max 
    
    plt.tight_layout()
    plt.show()

def extraire_morceau(fichier, duree_extrait):
    # Charger le fichier audio FLAC
    audio, frequence_echantillonnage = sf.read(fichier)
    # Calculer le nombre d'échantillons nécessaires pour l'extrait
    nombre_echantillons_extrait = int(duree_extrait * frequence_echantillonnage)
    # Extraire l'extrait du fichier audio
    extrait = audio[:nombre_echantillons_extrait]
    # Obtenir le nom du nouveau fichier avec l'extrait
    nom_nouveau_fichier = "extrait_" + str(duree_extrait) + "s_" + fichier
    # Écrire l'extrait dans un nouveau fichier FLAC
    sf.write(nom_nouveau_fichier, extrait, frequence_echantillonnage)
    print(f"Extrait de {duree_extrait} secondes du fichier audio a été extrait dans le fichier : {nom_nouveau_fichier}")

    return extrait, frequence_echantillonnage

def lire_piste(chemin_piste):
    audio_data, sample_rate = sf.read(chemin_piste)
    return audio_data, sample_rate

def sous_echantillonage(fic, echant, facteur):
    s_echant = echant / facteur
    sf.write(fic, echant, s_echant)
    return s_echant


def uniform_quantizer(s, niv, nmin, nmax):
    ''' quantification uniforme 
    sq, error = uniform_quantizer(s, niv, nmin, nmax):
    
    input : 
    - s : signal à quantifier
    - niv : nombre de niveaux de sortie
    - nmin : niveau minimal de sortie 
    - nmax : niveau maximal de sortie
    output : 
    - sq : signal quantifie
    - d: pas de quantification
    '''
    sq = empty(len(s))
    d = (nmax-nmin)/(niv-1)
    for i in range(len(s)):
        if s[i]>=nmax: 
            sq[i]=nmax
        elif s[i]<=nmin:
            sq[i]=nmin
        else :
            if niv % 2 == 0 :
                sq[i] = d * np.round((s[i]-d/2)/d) + (d/2)   
            else : 
                sq[i] = d * np.round(s[i]/d)        
    return sq, d

def ecrire_format_FLAC(pistes, sortie, frequence_echantillonnage):
    if len(pistes) != len(sortie):
        raise ValueError("Le nombre de pistes ne correspond pas au nombre de chemins de sortie.")

    for i in range(len(pistes)):
        piste = pistes[i]
        nom_sortie = sortie[i]

        # Normaliser la piste audio entre -1 et 1
        piste_normalisee = piste / np.max(np.abs(piste))

        # Vérifier la dimension de la piste
        if len(piste_normalisee.shape) > 1:
            piste_normalisee = np.squeeze(piste_normalisee)

        # Écrire la piste au format FLAC
        sf.write(nom_sortie, piste_normalisee, frequence_echantillonnage)

def fusionner_pistes(piste1, piste2, chemin_sortie, freq):
    # Concatenate the audio data of the two tracks
    piste_fusionnee = np.concatenate((piste1, piste2), axis=0)

    # Write the merged track to a FLAC file
    sf.write(chemin_sortie, piste_fusionnee, freq)

    print("Fusion des pistes terminée. Le résultat est enregistré dans le fichier :", chemin_sortie)

def quantif(nom_fichier, bits):
    duree_extrait = 5
    # Extraire l'extrait du fichier audio
    extrait_audio, frequence_echantillonnage = extraire_morceau(
        nom_fichier, duree_extrait)
    cara = obtenir_caracteristiques_audio(nom_fichier)
    pistes = cara["nombre_pistes"]
    piste1_audio = extrait_audio[:, 0] if pistes > 1 else extrait_audio
    piste2_audio = extrait_audio[:, 1]
    
    quant1 = uniform_quantizer(piste1_audio,bits,-1,1)
    quant2 = uniform_quantizer(piste2_audio,bits,-1,1)
    return quant1 , quant2, frequence_echantillonnage