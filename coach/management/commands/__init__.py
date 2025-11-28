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
            }
        ]

        # Create all drills
        all_drills = pronunciation_drills + filler_drills + pacing_drills
        
        for drill_data in all_drills:
            drill = Drill.objects.create(**drill_data)
            self.stdout.write(
                self.style.SUCCESS(f'Created drill: {drill.name}')
            )

        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {len(all_drills)} drills!')
        )
