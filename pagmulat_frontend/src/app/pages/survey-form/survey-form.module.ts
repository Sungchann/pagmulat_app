import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule } from '@angular/forms';
import { SurveyFormComponent } from './survey-form.component';

@NgModule({
  declarations: [SurveyFormComponent],
  imports: [CommonModule, ReactiveFormsModule],
  exports: [SurveyFormComponent]
})
export class SurveyFormModule {}
