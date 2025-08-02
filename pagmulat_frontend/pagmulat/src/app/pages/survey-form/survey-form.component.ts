import { Component, Output, EventEmitter } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormArray } from '@angular/forms';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-survey-form',
  templateUrl: './survey-form.component.html',
  styleUrls: ['./survey-form.component.css']
})
export class SurveyFormComponent {
  @Output() close = new EventEmitter<void>();

  surveyForm: FormGroup;
  isSubmitting = false;
  predictionResult: any = null;
  error: string | null = null;

  constructor(private fb: FormBuilder, private apiService: ApiService) {
    this.surveyForm = this.fb.group({
      year_level: ['', Validators.required],
      program: ['', Validators.required],
      age: ['', [Validators.required, Validators.min(15), Validators.max(99)]],
      gender: ['', Validators.required],
      lms_usage: ['', Validators.required],
      code_platform_usage: ['', Validators.required],
      ai_usage: ['', Validators.required],
      social_media_usage: ['', Validators.required],
      fixed_study_schedule: ['', Validators.required],
      study_hours: ['', Validators.required],
      study_start_time: ['', Validators.required],
      assignment_timeliness: ['', Validators.required],
      collab_tools_usage: ['', Validators.required],
      sleep_time: ['', Validators.required],
      burnout: ['', Validators.required],
      study_breaks: ['', Validators.required],
      motivation: ['', Validators.required],
      motivation_triggers: this.fb.array([]),
      productivity: ['', Validators.required],
      productivity_tools: this.fb.array([]),
      social_media_distraction: ['', Validators.required],
      distraction_platform: ['', Validators.required],
      block_distractions: ['', Validators.required],
      academic_tools: this.fb.array([]),
      digital_reliance: ['', Validators.required],
      // Optional fields
      digital_habits: [''],
      digital_change: [''],
      digital_reflection: ['']
    });
  }

  get motivation_triggers() { return this.surveyForm.get('motivation_triggers') as FormArray; }
  get productivity_tools() { return this.surveyForm.get('productivity_tools') as FormArray; }
  get academic_tools() { return this.surveyForm.get('academic_tools') as FormArray; }

  onCheckboxChange(event: any, formArrayName: string) {
    const formArray: FormArray = this.surveyForm.get(formArrayName) as FormArray;
    if (event.target.checked) {
      formArray.push(this.fb.control(event.target.value));
    } else {
      const idx = formArray.controls.findIndex(x => x.value === event.target.value);
      if (idx !== -1) formArray.removeAt(idx);
    }
  }

  submitSurvey() {
    this.isSubmitting = true;
    this.error = null;
    this.predictionResult = null;
    this.apiService.predictProductivity(this.surveyForm.value).subscribe({
      next: (result) => {
        this.predictionResult = result;
        this.isSubmitting = false;
      },
      error: (err) => {
        this.error = 'Prediction failed.';
        this.isSubmitting = false;
      }
    });
  }

  onClose() {
    this.close.emit();
  }
}
