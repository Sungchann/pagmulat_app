import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AnalysisComponent } from './analysis.component';
import { ParameterFormComponent } from './parameter-form/parameter-form.component';
import { VisualizationComponent } from './visualization/visualization.component';
import { SharedModule } from '../../shared/shared.module';

@NgModule({
  declarations: [
    AnalysisComponent,
    ParameterFormComponent,
    VisualizationComponent
  ],
  imports: [
    CommonModule,
    SharedModule  // Import shared components
  ]
})
export class AnalysisModule { }