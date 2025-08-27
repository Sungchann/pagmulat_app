import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
// import { DashboardComponent } from './pages/dashboard/dashboard.component';
import { DashboardComponent } from './features/dashboard/dashboard.component';
import { PipelineComponent } from './features/pipeline/pipeline.component';
import { AlgorithmComparisonComponent } from './features/algorithm-comparison/algorithm-comparison.component';

const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: DashboardComponent},
  { path: 'pipeline', component: PipelineComponent},
  { path: 'comparison', component: AlgorithmComparisonComponent}
  // { path: 'analysis', loadChildren: () => import('./pages/analysis/analysis.module').then(m => m.AnalysisModule) },
  // { path: 'upload', loadChildren: () => import('./pages/upload/upload.module').then(m => m.UploadModule) },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
