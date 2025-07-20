import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './pages/dashboard/dashboard.component';

const routes: Routes = [
  { path: '', component: DashboardComponent },
  // { path: 'analysis', loadChildren: () => import('./pages/analysis/analysis.module').then(m => m.AnalysisModule) },
  // { path: 'upload', loadChildren: () => import('./pages/upload/upload.module').then(m => m.UploadModule) },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
