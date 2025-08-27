
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { SharedModule } from './shared/shared.module';
import { AppComponent } from './app.component';
// import { DashboardModule } from './pages/dashboard/dashboard.module';
import { HttpClientModule } from '@angular/common/http';
import { SidebarComponent } from './shared-components/sidebar/sidebar.component';
import { DashboardComponent } from './features/dashboard/dashboard.component';
import { PipelineComponent } from './features/pipeline/pipeline.component';
import { AlgorithmComparisonComponent } from './features/algorithm-comparison/algorithm-comparison.component';
import { DashboardModule } from './features/dashboard/dashboard.module';

@NgModule({
  declarations: [AppComponent, SidebarComponent, PipelineComponent, AlgorithmComparisonComponent],
  imports: [BrowserModule, AppRoutingModule, SharedModule, DashboardModule, HttpClientModule],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
