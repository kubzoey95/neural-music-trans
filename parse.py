from music21.converter import parse
from music21.chord import Chord
from music21.note import Note
from glob import iglob
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pyblaze.multiprocessing as xmp
from matplotlib import pyplot
from music21 import *
import os


def unpack(chord_note):
	try:
		if isinstance(chord_note, Chord):
			return [pitch.midi for pitch in chord_note.pitches]
		return [chord_note.pitch.midi]
	except:
		return []

def convertQuarter(q):
	return round(q * 100)


def mid_file_to_sequence(path):
	print(f"Processing {path}")
	try:
		mid = parse(path)
	except:
		return
	all_notes = [(pitch, note.offset) for note in sorted(mid.flat.notes, key=lambda x: x.offset) for pitch in unpack(note)]
	array = np.array([(all_notes[0][0], convertQuarter(all_notes[0][1])), *((pitch, convertQuarter(tick - all_notes[i][1])) for i, (pitch, tick) in enumerate(all_notes[1:]))])
	return array

def convert_offsets_to_waits(inp):
	events = []
	for arr in inp:
		if arr[1] != 0:
			events.append((arr[1], 1))
		events.append((arr[0], 0))
	return np.array(events)

def create_dataset(path="dataset/Bach/", save_as="dataset_bach.npz"):
	tokenizer = xmp.Vectorizer(mid_file_to_sequence, num_workers=4)
	sequences = tokenizer.process((mid for mid in iglob(os.path.join(path, "**/*.mid"), recursive=True)))
	np.savez_compressed(save_as, *(sequence for sequence in sequences if sequence is not None))

possible_components = np.array([2, 3, 5, 7, 8, 10, 25, 30, 40, 50, 60, 70, 100, 150, 200, 500])

components_dict = {comp: i for i, comp in enumerate(possible_components)}

def split_number_into_components(number):
	smaller = possible_components[possible_components <= number]
	rest = number
	
	number_set = []
	
	for num in smaller[::-1]:
		if num <= rest:
			times = rest // num
			number_set.extend([num] * times)
			rest -= num * times
			if rest == 0:
				break
	return number_set


def get_score_with_components(inp):
	return np.array([event for arr in inp for event in (([num, 1] for num in split_number_into_components(arr[0])) if arr[1] == 1 else [arr])])


def tokenize_score(inp, seq_len=1024):
	tokens = inp[:, 0]
	tokens[inp[:, 1] == 1] = np.vectorize(components_dict.get)(tokens[inp[:, 1] == 1]) + 128
	tokens += 1
	zeros = np.zeros(seq_len)
	tokens = np.concatenate((zeros, tokens, np.zeros(1)))
	return tokens