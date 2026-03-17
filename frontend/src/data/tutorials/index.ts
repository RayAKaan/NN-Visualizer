import type { Lesson } from "../../types/tutorial";
import { LESSON_FORWARD_PASS } from "./lesson01-forward-pass";
import { LESSON_BACKPROP } from "./lesson02-backprop";
import { LESSON_CNN } from "./lesson03-cnn";

export const LESSONS: Lesson[] = [LESSON_FORWARD_PASS, LESSON_BACKPROP, LESSON_CNN];
