{{/* Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0 */}}

{{- define "keystone.name" -}}
keystone
{{- end -}}

{{- define "keystone.fullname" -}}
{{- .Release.Name -}}
{{- end -}}

{{- define "keystone.labels" -}}
app.kubernetes.io/name: {{ include "keystone.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "keystone.selectorLabels" -}}
app.kubernetes.io/name: {{ include "keystone.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Renders a full image reference. Call as:
  {{ include "keystone.image" (dict "root" $ "image" .Values.image.app) }}
(a plain `include "keystone.image" .Values.image.app` cannot see
.Values.global from inside the sub-scope, so the root context must be
passed through explicitly.)
*/}}
{{- define "keystone.image" -}}
{{- $registry := .root.Values.global.imageRegistry -}}
{{- if $registry -}}
{{ $registry }}/{{ .image.repository }}:{{ .image.tag }}
{{- else -}}
{{ .image.repository }}:{{ .image.tag }}
{{- end -}}
{{- end -}}
