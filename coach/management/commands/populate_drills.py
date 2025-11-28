from django.core.management.base import BaseCommand
from coach.models import Drill


class Command(BaseCommand):
    help = 'Populate the database with sample pronunciation drills'

    def handle(self, *args, **options):
        # Clear existing drills
        Drill.objects.all().delete()
        
        # Create pronunciation drills
        pronunciation_drills = [
            {
                'name': 'Poem Echo Challenge',
                'skill_type': 'pronunciation',
                'cause': 'poor_skills',
                'description': 'Practice pronunciation by reading MLK excerpts with phonetic guidance and audio feedback.',
                'interactive_elements': {
                    'timer_duration': 300,
                    'prompts': ['Read the excerpt slowly', 'Focus on difficult words', 'Record and compare'],
                    'audio_features': True,
                    'phonetic_tips': True
                }
            },
            {
                'name': 'Exaggerated Vowel Vortex',
                'skill_type': 'pronunciation',
                'cause': 'poor_skills',
                'description': 'Improve vowel pronunciation through exaggerated articulation exercises with spinning wheel selection.',
                'interactive_elements': {
                    'timer_duration': 180,
                    'prompts': ['Spin for a vowel', 'Say 5 words exaggeratedly', 'Focus on mouth position'],
                    'spinning_wheel': True,
                    'vowel_words': True
                }
            },
            {
                'name': 'Mirror Mimic Madness',
                'skill_type': 'pronunciation',
                'cause': 'lack_of_confidence',
                'description': 'Build confidence and articulation through tongue twisters with webcam mirror feedback.',
                'interactive_elements': {
                    'timer_duration': 240,
                    'prompts': ['Watch yourself speak', 'Repeat 3 times', 'Focus on mouth movements'],
                    'webcam_mirror': True,
                    'tongue_twisters': True
                }
            },
            {
                'name': 'Phonetic Puzzle Builder',
                'skill_type': 'pronunciation',
                'cause': 'poor_skills',
                'description': 'Learn phonetic symbols through drag-and-drop puzzle building with audio verification.',
                'interactive_elements': {
                    'timer_duration': 300,
                    'prompts': ['Drag symbols to spell', 'Record your pronunciation', 'Check against answer'],
                    'drag_drop': True,
                    'phonetic_symbols': True
                }
            },
            {
                'name': 'Shadow Superhero',
                'skill_type': 'pronunciation',
                'cause': 'poor_skills',
                'description': 'Improve pronunciation and pacing by shadowing TED talk excerpts with waveform comparison.',
                'interactive_elements': {
                    'timer_duration': 300,
                    'prompts': ['Listen to reference', 'Shadow the speaker', 'Compare waveforms'],
                    'reference_audio': True,
                    'waveform_comparison': True
                }
            },
            {
                'name': 'Pencil Precision Drill',
                'skill_type': 'pronunciation',
                'cause': 'poor_skills',
                'description': 'Enhance articulation clarity by speaking with a pencil between teeth, comparing before/after.',
                'interactive_elements': {
                    'timer_duration': 180,
                    'prompts': ['Record without pencil', 'Record with pencil', 'Compare clarity scores'],
                    'comparison_test': True,
                    'clarity_scoring': True
                }
            }
        ]

        # Create filler word drills
        filler_drills = [
            {
                'name': 'Filler Zap Game',
                'skill_type': 'filler_words',
                'cause': 'anxiety',
                'description': 'Eliminate filler words through interactive games and awareness exercises.',
                'interactive_elements': {
                    'timer_duration': 120,
                    'prompts': ['Identify filler words', 'Practice pauses', 'Build awareness'],
                    'filler_detection': True,
                    'pause_practice': True
                }
            },
            {
                'name': 'Silence Master',
                'skill_type': 'filler_words',
                'cause': 'stress',
                'description': 'Learn to embrace comfortable silence instead of using filler words.',
                'interactive_elements': {
                    'timer_duration': 180,
                    'prompts': ['Practice comfortable silence', 'Count silent seconds', 'Build confidence'],
                    'silence_timer': True,
                    'confidence_building': True
                }
            }
        ]

        # Create pacing drills
        pacing_drills = [
            {
                'name': 'Rhythm Ruler',
                'skill_type': 'pacing',
                'cause': 'anxiety',
                'description': 'Develop consistent speaking rhythm through metronome-guided practice.',
                'interactive_elements': {
                    'timer_duration': 240,
                    'prompts': ['Follow the beat', 'Maintain steady pace', 'Practice rhythm'],
                    'metronome': True,
                    'rhythm_training': True
                }
            },
            {
                'name': 'Speed Controller',
                'skill_type': 'pacing',
                'cause': 'stress',
                'description': 'Learn to control speaking speed through gradual pace adjustment exercises.',
                'interactive_elements': {
                    'timer_duration': 300,
                    'prompts': ['Start slow', 'Gradually increase', 'Find your pace'],
                    'speed_control': True,
                    'pace_adjustment': True
                }
            },
            # New Pacing Drills
            {
                'name': 'Metronome Rhythm Race',
                'skill_type': 'pacing',
                'cause': 'poor_skills',
                'description': 'Race against virtual opponents while maintaining perfect pacing with metronome guidance.',
                'interactive_elements': {
                    'metronome': True,
                    'opponent_levels': 4,
                    'wpm_target': 150,
                    'race_tracking': True
                }
            },
            {
                'name': 'Pause Pyramid Builder',
                'skill_type': 'pacing',
                'cause': 'anxiety',
                'description': 'Build your speech with strategic pauses at different levels using a pyramid structure.',
                'interactive_elements': {
                    'pyramid_levels': 5,
                    'pause_timer': True,
                    'visual_feedback': True,
                    'progressive_difficulty': True
                }
            },
            {
                'name': 'Slow-Motion Story Slam',
                'skill_type': 'pacing',
                'cause': 'stress',
                'description': 'Practice speaking at half-speed with adjustable playback and plot twists.',
                'interactive_elements': {
                    'speed_slider': True,
                    'story_prompts': True,
                    'plot_twists': True,
                    'playback_control': True
                }
            },
            {
                'name': 'Beat Drop Dialogue',
                'skill_type': 'pacing',
                'cause': 'poor_skills',
                'description': 'Sync your speech to rhythmic beats with strategic pause placement.',
                'interactive_elements': {
                    'beat_patterns': True,
                    'rhythmic_beats': True,
                    'drop_sync': True,
                    'alignment_scoring': True
                }
            },
            {
                'name': 'Timer Tag Team',
                'skill_type': 'pacing',
                'cause': 'anxiety',
                'description': 'Practice alternating speech and pause cycles with partner prompts.',
                'interactive_elements': {
                    'alternating_timers': True,
                    'partner_prompts': True,
                    'buzzers': True,
                    'cycle_tracking': True
                }
            },
            {
                'name': 'Echo Chamber Escalation',
                'skill_type': 'pacing',
                'cause': 'stress',
                'description': 'Match varying speech speeds with escalating difficulty levels.',
                'interactive_elements': {
                    'echo_playback': True,
                    'speed_matching': True,
                    'difficulty_levels': 5,
                    'escalation_system': True
                }
            }
        ]

        # New Filler Words Drills
        new_filler_drills = [
            {
                'name': 'Filler Zap Game',
                'skill_type': 'filler_words',
                'cause': 'poor_skills',
                'description': 'Zap filler words in real-time with streak tracking and scoring.',
                'interactive_elements': {
                    'zap_buttons': True,
                    'real_time_detection': True,
                    'streak_tracking': True,
                    'scoring_system': True
                }
            },
            {
                'name': 'Silent Switcheroo',
                'skill_type': 'filler_words',
                'cause': 'anxiety',
                'description': 'Rewrite sentences to eliminate fillers and practice cleaner speech.',
                'interactive_elements': {
                    'text_rewrite': True,
                    'filler_highlighting': True,
                    'comparison': True,
                    'improvement_tracking': True
                }
            },
            {
                'name': 'Pause Power-Up',
                'skill_type': 'filler_words',
                'cause': 'stress',
                'description': 'Replace fillers with strategic pauses in timed sessions.',
                'interactive_elements': {
                    'pause_replacement': True,
                    'power_levels': 5,
                    'progress_bar': True,
                    'clean_run_tracking': True
                }
            },
            {
                'name': 'Filler Hunt Mirror Maze',
                'skill_type': 'filler_words',
                'cause': 'lack_of_confidence',
                'description': 'Use webcam mirror to hunt and eliminate fillers in real-time.',
                'interactive_elements': {
                    'webcam_mirror': True,
                    'real_time_hunting': True,
                    'flash_alerts': True,
                    'mid_sentence_correction': True
                }
            },
            {
                'name': 'Word Swap Whirlwind',
                'skill_type': 'filler_words',
                'cause': 'poor_skills',
                'description': 'Spin wheel for filler replacements and practice impromptu speech.',
                'interactive_elements': {
                    'spinning_wheel': True,
                    'word_swapping': True,
                    'impromptu_speech': True,
                    'mic_grading': True
                }
            },
            {
                'name': 'Echo Elimination Echo',
                'skill_type': 'filler_words',
                'cause': 'anxiety',
                'description': 'Practice with echo playback that omits fillers for cleaner repetition.',
                'interactive_elements': {
                    'echo_playback': True,
                    'filler_elimination': True,
                    'similarity_rating': True,
                    'playback_controls': True
                }
            }
        ]

        # Create all drills
        all_drills = pronunciation_drills + filler_drills + pacing_drills + new_filler_drills
        
        for drill_data in all_drills:
            drill = Drill.objects.create(**drill_data)
            self.stdout.write(
                self.style.SUCCESS(f'Created drill: {drill.name}')
            )

        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {len(all_drills)} drills!')
        )
