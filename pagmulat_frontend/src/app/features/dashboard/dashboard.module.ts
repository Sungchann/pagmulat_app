import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SharedModule } from '../../shared/shared.module';
import { SurveyFormModule } from 'src/app/pages/survey-form/survey-form.module';
import { DashboardComponent } from './dashboard.component';

@NgModule({
  declarations: [DashboardComponent],
  imports: [CommonModule, SharedModule, SurveyFormModule],
  exports: [DashboardComponent]
})
export class DashboardModule {}
