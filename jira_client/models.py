from __future__ import annotations
from pydantic import BaseModel, Field
from uuid import UUID
from typing import Any
from enum import Enum
from datetime import date, datetime

__all__ = (
    'ActorInputBean',
    'ActorsMap',
    'AddAtlassianTeamRequest',
    'AddAtlassianTeamRequestPlanningStyle',
    'AddFieldBean',
    'AddGroupBean',
    'AddNotificationsDetails',
    'AddSecuritySchemeLevelsRequestBean',
    'AnnouncementBannerConfiguration',
    'AnnouncementBannerConfigurationUpdate',
    'AnnouncementBannerConfigurationVisibility',
    'AppWorkflowTransitionRule',
    'AppWorkflowTransitionRuleUnnamedModel',
    'Application',
    'ApplicationProperty',
    'ApplicationRole',
    'ApprovalConfiguration',
    'ApprovalConfigurationConditionType',
    'ApprovalConfigurationPreview',
    'ArchiveIssueAsyncRequest',
    'ArchivedIssuesFilterRequest',
    'AssociateFieldConfigurationsWithIssueTypesRequest',
    'AssociateSecuritySchemeWithProjectDetails',
    'AssociatedItemBean',
    'AssociationContextObject',
    'Attachment',
    'AttachmentArchiveEntry',
    'AttachmentArchiveImpl',
    'AttachmentArchiveItemReadable',
    'AttachmentArchiveMetadataReadable',
    'AttachmentMetadata',
    'AttachmentMetadataAccountType',
    'AttachmentMetadataUnnamedModel',
    'AttachmentMetadataUnnamedModel1',
    'AttachmentMetadataUnnamedModel2',
    'AttachmentSettings',
    'AttachmentUnnamedModel',
    'AttachmentUnnamedModel1',
    'AttachmentUserDetails',
    'AuditRecordBean',
    'AuditRecords',
    'AutoCompleteSuggestion',
    'AutoCompleteSuggestions',
    'AutoEnum',
    'AutoEnum1',
    'AutoEnum10',
    'AutoEnum11',
    'AutoEnum12',
    'AutoEnum13',
    'AutoEnum14',
    'AutoEnum15',
    'AutoEnum16',
    'AutoEnum17',
    'AutoEnum18',
    'AutoEnum19',
    'AutoEnum2',
    'AutoEnum20',
    'AutoEnum21',
    'AutoEnum22',
    'AutoEnum23',
    'AutoEnum24',
    'AutoEnum25',
    'AutoEnum26',
    'AutoEnum27',
    'AutoEnum28',
    'AutoEnum29',
    'AutoEnum3',
    'AutoEnum30',
    'AutoEnum31',
    'AutoEnum32',
    'AutoEnum33',
    'AutoEnum34',
    'AutoEnum4',
    'AutoEnum5',
    'AutoEnum6',
    'AutoEnum7',
    'AutoEnum8',
    'AutoEnum9',
    'AvailableDashboardGadget',
    'AvailableDashboardGadgetsResponse',
    'AvailableWorkflowConnectRule',
    'AvailableWorkflowConnectRuleRuleType',
    'AvailableWorkflowForgeRule',
    'AvailableWorkflowSystemRule',
    'AvailableWorkflowTriggerTypes',
    'AvailableWorkflowTriggers',
    'Avatar',
    'AvatarUrlsBean',
    'Avatars',
    'BoardColumnPayload',
    'BoardFeaturePayload',
    'BoardFeaturePayloadFeatureKey',
    'BoardFeaturePayloadState',
    'BoardFeaturesPayload',
    'BoardPayload',
    'BoardPayloadCardColorStrategy',
    'BoardsPayload',
    'BulkChangeOwnerDetails',
    'BulkChangelogRequestBean',
    'BulkChangelogResponseBean',
    'BulkContextualConfiguration',
    'BulkCustomFieldOptionCreateRequest',
    'BulkCustomFieldOptionUpdateRequest',
    'BulkEditActionError',
    'BulkEditGetFields',
    'BulkEditShareableEntityRequest',
    'BulkEditShareableEntityRequestAction',
    'BulkEditShareableEntityRequestUnnamedModel',
    'BulkEditShareableEntityRequestUnnamedModel1',
    'BulkEditShareableEntityResponse',
    'BulkFetchIssueRequestBean',
    'BulkIssueIsWatching',
    'BulkIssuePropertyUpdateRequest',
    'BulkIssuePropertyUpdateRequestUnnamedModel',
    'BulkIssueResults',
    'BulkOperationErrorResponse',
    'BulkOperationErrorResult',
    'BulkOperationProgress',
    'BulkOperationProgressStatus',
    'BulkPermissionGrants',
    'BulkPermissionsRequestBean',
    'BulkProjectPermissionGrants',
    'BulkProjectPermissions',
    'BulkRedactionRequest',
    'BulkRedactionResponse',
    'BulkTransitionGetAvailableTransitions',
    'BulkTransitionSubmitInput',
    'BulkWorklogKeyRequestBean',
    'BulkWorklogKeyResponseBean',
    'CardLayout',
    'CardLayoutField',
    'CardLayoutFieldMode',
    'ChangeDetails',
    'ChangeFilterOwner',
    'ChangedValueBean',
    'ChangedWorklog',
    'ChangedWorklogs',
    'Changelog',
    'ChangelogUnnamedModel',
    'ChangelogUnnamedModel1',
    'ChangelogUnnamedModel2',
    'ChangelogUnnamedModel3',
    'ChangelogUnnamedModel4',
    'ChangelogUnnamedModel5',
    'ColumnItem',
    'ColumnRequestBody',
    'Comment',
    'CommentType',
    'CommentUnnamedModel',
    'CommentUnnamedModel1',
    'CommentUnnamedModel2',
    'CommentUnnamedModel3',
    'CommentUnnamedModel4',
    'CommentUserDetails',
    'CommentUserDetails1',
    'ComponentIssuesCount',
    'ComponentJsonBean',
    'ComponentWithIssueCount',
    'ComponentWithIssueCountUnnamedModel',
    'ComponentWithIssueCountUnnamedModel1',
    'ComponentWithIssueCountUnnamedModel2',
    'ComponentWithIssueCountUnnamedModel3',
    'ComponentWithIssueCountUnnamedModel4',
    'ComponentWithIssueCountUnnamedModel5',
    'ComponentWithIssueCountUnnamedModel6',
    'ComponentWithIssueCountUnnamedModel7',
    'ComponentWithIssueCountUnnamedModel8',
    'ComponentWithIssueCountUser',
    'ComponentWithIssueCountUser1',
    'ComponentWithIssueCountUser2',
    'CompoundClause',
    'CompoundClauseOperator',
    'ConditionGroupConfiguration',
    'ConditionGroupPayload',
    'ConditionGroupPayloadOperation',
    'ConditionGroupUpdate',
    'Configuration',
    'ConfigurationDefaultUnit',
    'ConfigurationTimeFormat',
    'ConfigurationUnnamedModel',
    'ConfigurationsListParameters',
    'ConnectCustomFieldValue',
    'ConnectCustomFieldValueType',
    'ConnectCustomFieldValues',
    'ConnectModules',
    'ContainerForProjectFeatures',
    'ContainerForRegisteredWebhooks',
    'ContainerForWebhookIDs',
    'ContainerOfWorkflowSchemeAssociations',
    'ContentItem',
    'ContentItemEntityType',
    'Context',
    'ContextForProjectAndIssueType',
    'ContextProjectDetails',
    'ContextScope',
    'ContextUnnamedModel',
    'ContextUnnamedModel1',
    'ContextUnnamedModel2',
    'ContextUnnamedModel3',
    'ContextualConfiguration',
    'ConvertedJQLQueries',
    'CreateCrossProjectReleaseRequest',
    'CreateCustomFieldContext',
    'CreateCustomFieldRequest',
    'CreateDateFieldRequest',
    'CreateExclusionRulesRequest',
    'CreateFieldAssociationSchemeLinksBean',
    'CreateFieldAssociationSchemeRequest',
    'CreateFieldAssociationSchemeResponse',
    'CreateIssueSecuritySchemeDetails',
    'CreateIssueSourceRequest',
    'CreateIssueSourceRequestType',
    'CreateNotificationSchemeDetails',
    'CreatePermissionHolderRequest',
    'CreatePermissionRequest',
    'CreatePermissionRequestType',
    'CreatePermissionRequestType1',
    'CreatePermissionRequestUnnamedModel',
    'CreatePlanOnlyTeamRequest',
    'CreatePlanRequest',
    'CreatePlanRequestDependencies',
    'CreatePlanRequestEstimation',
    'CreatePlanRequestInferredDates',
    'CreatePlanRequestType',
    'CreatePlanRequestUnnamedModel',
    'CreatePlanRequestUnnamedModel1',
    'CreatePlanRequestUnnamedModel2',
    'CreatePlanRequestUnnamedModel3',
    'CreatePriorityDetails',
    'CreatePriorityDetailsIconUrl',
    'CreatePrioritySchemeDetails',
    'CreatePrioritySchemeDetailsUnnamedModel',
    'CreateProjectDetails',
    'CreateProjectDetailsProjectTemplateKey',
    'CreateResolutionDetails',
    'CreateSchedulingRequest',
    'CreateUiModificationDetails',
    'CreateUpdateRoleRequestBean',
    'CreatedIssue',
    'CreatedIssueUnnamedModel',
    'CreatedIssueUnnamedModel1',
    'CreatedIssues',
    'CustomFieldConfigurations',
    'CustomFieldContext',
    'CustomFieldContextDefaultValueCascadingOption',
    'CustomFieldContextDefaultValueDate',
    'CustomFieldContextDefaultValueDateTime',
    'CustomFieldContextDefaultValueFloat',
    'CustomFieldContextDefaultValueForgeDateTimeField',
    'CustomFieldContextDefaultValueForgeGroupField',
    'CustomFieldContextDefaultValueForgeMultiGroupField',
    'CustomFieldContextDefaultValueForgeMultiStringField',
    'CustomFieldContextDefaultValueForgeMultiUserField',
    'CustomFieldContextDefaultValueForgeNumberField',
    'CustomFieldContextDefaultValueForgeObjectField',
    'CustomFieldContextDefaultValueForgeStringField',
    'CustomFieldContextDefaultValueForgeUserField',
    'CustomFieldContextDefaultValueLabels',
    'CustomFieldContextDefaultValueMultiUserPicker',
    'CustomFieldContextDefaultValueMultipleGroupPicker',
    'CustomFieldContextDefaultValueMultipleOption',
    'CustomFieldContextDefaultValueMultipleVersionPicker',
    'CustomFieldContextDefaultValueProject',
    'CustomFieldContextDefaultValueReadOnly',
    'CustomFieldContextDefaultValueSingleGroupPicker',
    'CustomFieldContextDefaultValueSingleOption',
    'CustomFieldContextDefaultValueSingleVersionPicker',
    'CustomFieldContextDefaultValueTextArea',
    'CustomFieldContextDefaultValueTextField',
    'CustomFieldContextDefaultValueURL',
    'CustomFieldContextDefaultValueUpdate',
    'CustomFieldContextOption',
    'CustomFieldContextProjectMapping',
    'CustomFieldContextSingleUserPickerDefaults',
    'CustomFieldContextUpdateDetails',
    'CustomFieldCreatedContextOptionsList',
    'CustomFieldDefinitionJsonBean',
    'CustomFieldDefinitionJsonBeanSearcherKey',
    'CustomFieldOption',
    'CustomFieldOptionCreate',
    'CustomFieldOptionUpdate',
    'CustomFieldReplacement',
    'CustomFieldUpdatedContextOptionsList',
    'CustomFieldValueUpdate',
    'CustomFieldValueUpdateDetails',
    'CustomTemplateOptions',
    'CustomTemplateRequestDTO',
    'CustomTemplatesProjectDetails',
    'CustomTemplatesProjectDetailsAccessLevel',
    'Dashboard',
    'DashboardDetails',
    'DashboardGadget',
    'DashboardGadgetColor',
    'DashboardGadgetPosition',
    'DashboardGadgetResponse',
    'DashboardGadgetSettings',
    'DashboardGadgetSettingsUnnamedModel',
    'DashboardGadgetUnnamedModel',
    'DashboardGadgetUpdateRequest',
    'DashboardGadgetUpdateRequestUnnamedModel',
    'DashboardUnnamedModel',
    'DashboardUnnamedModel1',
    'DashboardUserBean',
    'DataClassificationLevelsBean',
    'DataClassificationTagBean',
    'DateRangeFilterRequest',
    'DefaultLevelValue',
    'DefaultShareScope',
    'DefaultShareScopeScope',
    'DefaultWorkflow',
    'DefaultWorkflowEditorResponse',
    'DefaultWorkflowEditorResponseValue',
    'DeleteAndReplaceVersionBean',
    'DeleteFieldAssociationSchemeResponse',
    'DetailedErrorCollection',
    'DocumentVersion',
    'DuplicatePlanRequest',
    'EditTemplateRequest',
    'EntityProperty',
    'EntityPropertyDetails',
    'Error',
    'ErrorCollection',
    'ErrorMessage',
    'Errors',
    'EventNotification',
    'EventNotificationFieldDetails',
    'EventNotificationNotificationType',
    'EventNotificationProjectDetails',
    'EventNotificationProjectDetails1',
    'EventNotificationProjectRole',
    'EventNotificationScope',
    'EventNotificationScope1',
    'EventNotificationUnnamedModel',
    'EventNotificationUnnamedModel1',
    'EventNotificationUnnamedModel10',
    'EventNotificationUnnamedModel11',
    'EventNotificationUnnamedModel12',
    'EventNotificationUnnamedModel13',
    'EventNotificationUnnamedModel2',
    'EventNotificationUnnamedModel3',
    'EventNotificationUnnamedModel4',
    'EventNotificationUnnamedModel5',
    'EventNotificationUnnamedModel6',
    'EventNotificationUnnamedModel7',
    'EventNotificationUnnamedModel8',
    'EventNotificationUnnamedModel9',
    'EventNotificationUserDetails',
    'ExpandPrioritySchemePage',
    'ExportArchivedIssuesTaskProgressResponse',
    'FailedWebhook',
    'FailedWebhooks',
    'FieldAssociationItemPayload',
    'FieldAssociationParameters',
    'FieldAssociationSchemeFieldSearchResult',
    'FieldAssociationSchemeLinks',
    'FieldAssociationSchemeLinksBean',
    'FieldAssociationSchemeMatchedFilters',
    'FieldAssociationSchemeProjectSearchResult',
    'FieldAssociationsRequest',
    'FieldCapabilityPayload',
    'FieldChangedClause',
    'FieldChangedClauseOperator',
    'FieldConfiguration',
    'FieldConfigurationDetails',
    'FieldConfigurationIssueTypeItem',
    'FieldConfigurationItem',
    'FieldConfigurationItemsDetails',
    'FieldConfigurationScheme',
    'FieldConfigurationSchemeProjectAssociation',
    'FieldConfigurationSchemeProjects',
    'FieldConfigurationToIssueTypeMapping',
    'FieldCreateMetadata',
    'FieldCreateMetadataUnnamedModel',
    'FieldDetails',
    'FieldDetailsProjectDetails',
    'FieldDetailsScope',
    'FieldDetailsUnnamedModel',
    'FieldDetailsUnnamedModel1',
    'FieldDetailsUnnamedModel2',
    'FieldDetailsUnnamedModel3',
    'FieldDetailsUnnamedModel4',
    'FieldIdentifierObject',
    'FieldLastUsed',
    'FieldLastUsedType',
    'FieldLayoutSchemePayload',
    'FieldMetadata',
    'FieldMetadataUnnamedModel',
    'FieldProjectAssociation',
    'FieldReferenceData',
    'FieldSchemePayload',
    'FieldSchemePayloadOnConflict',
    'FieldSchemeToFieldsPartialFailure',
    'FieldSchemeToFieldsResponse',
    'FieldSchemeToProjectsPartialFailure',
    'FieldSchemeToProjectsRequest',
    'FieldSchemeToProjectsResponse',
    'FieldUpdateOperation',
    'FieldValueClause',
    'FieldValueClauseOperator',
    'FieldWasClause',
    'FieldWasClauseOperator',
    'Field_',
    'Fields',
    'FieldsSchemeItemParameter',
    'FieldsSchemeItemParameterRendererType',
    'FieldsSchemeItemWorkTypeParameter',
    'Filter',
    'FilterDetails',
    'FilterDetailsUnnamedModel',
    'FilterDetailsUnnamedModel1',
    'FilterDetailsUnnamedModel2',
    'FilterDetailsUser',
    'FilterSubscription',
    'FilterSubscriptionUnnamedModel',
    'FilterSubscriptionUnnamedModel1',
    'FilterSubscriptionUnnamedModel2',
    'FilterSubscriptionUnnamedModel3',
    'FilterSubscriptionUser',
    'FilterSubscriptionsList',
    'FilterUnnamedModel',
    'FilterUnnamedModel1',
    'FilterUnnamedModel2',
    'FilterUnnamedModel3',
    'FilterUnnamedModel4',
    'FilterUser',
    'ForgePanelProjectPinAsyncResponse',
    'ForgePanelProjectPinRequest',
    'FoundGroup',
    'FoundGroupManagedBy',
    'FoundGroupUsageType',
    'FoundGroups',
    'FoundUsers',
    'FoundUsersAndGroups',
    'FromLayoutPayload',
    'FunctionOperand',
    'FunctionReferenceData',
    'GetAtlassianTeamResponse',
    'GetCrossProjectReleaseResponse',
    'GetCustomFieldResponse',
    'GetDateFieldResponse',
    'GetExclusionRulesResponse',
    'GetFieldAssociationParametersResponse',
    'GetFieldAssociationSchemeByIdResponse',
    'GetFieldAssociationSchemeResponse',
    'GetIssueSourceResponse',
    'GetIssueSourceResponseType',
    'GetPermissionHolderResponse',
    'GetPermissionResponse',
    'GetPermissionResponseUnnamedModel',
    'GetPlanOnlyTeamResponse',
    'GetPlanResponse',
    'GetPlanResponseForPage',
    'GetPlanResponseForPageStatus',
    'GetPlanResponseUnnamedModel',
    'GetPlanResponseUnnamedModel1',
    'GetPlanResponseUnnamedModel2',
    'GetPlanResponseUnnamedModel3',
    'GetProjectsWithFieldSchemesResponse',
    'GetSchedulingResponse',
    'GetTeamResponseForPage',
    'GetTeamResponseForPageType',
    'GlobalScopeBean',
    'Group',
    'GroupDetails',
    'GroupLabel',
    'GroupLabelType',
    'GroupName',
    'GroupUnnamedModel',
    'HealthCheckResult',
    'Hierarchy',
    'HistoryMetadata',
    'HistoryMetadataParticipant',
    'Icon',
    'IdBean',
    'IdOrKeyBean',
    'IncludedFields',
    'IssueArchivalSyncRequest',
    'IssueArchivalSyncResponse',
    'IssueBean',
    'IssueBeanUnnamedModel',
    'IssueBeanUnnamedModel1',
    'IssueBeanUnnamedModel2',
    'IssueBulkDeletePayload',
    'IssueBulkEditField',
    'IssueBulkEditFieldEnum',
    'IssueBulkEditPayload',
    'IssueBulkEditPayloadUnnamedModel',
    'IssueBulkEditPayloadUnnamedModel1',
    'IssueBulkEditPayloadUnnamedModel2',
    'IssueBulkEditPayloadUnnamedModel3',
    'IssueBulkEditPayloadUnnamedModel4',
    'IssueBulkEditPayloadUnnamedModel5',
    'IssueBulkMovePayload',
    'IssueBulkTransitionForWorkflow',
    'IssueBulkTransitionPayload',
    'IssueBulkWatchOrUnwatchPayload',
    'IssueChangeLog',
    'IssueChangelogIds',
    'IssueCommentListRequestBean',
    'IssueContextVariable',
    'IssueCreateMetadata',
    'IssueEntityProperties',
    'IssueEntityPropertiesForMultiUpdate',
    'IssueError',
    'IssueEvent',
    'IssueFieldOption',
    'IssueFieldOptionConfiguration',
    'IssueFieldOptionConfigurationEnum',
    'IssueFieldOptionConfigurationUnnamedModel',
    'IssueFieldOptionConfigurationUnnamedModel1',
    'IssueFieldOptionCreateBean',
    'IssueFieldOptionScopeBean',
    'IssueFilterForBulkPropertyDelete',
    'IssueFilterForBulkPropertySet',
    'IssueLimitReportResponseBean',
    'IssueLink',
    'IssueLinkFields',
    'IssueLinkFields1',
    'IssueLinkIssueTypeDetails',
    'IssueLinkIssueTypeDetails1',
    'IssueLinkLinkedIssue',
    'IssueLinkLinkedIssue1',
    'IssueLinkPriority',
    'IssueLinkPriority1',
    'IssueLinkProjectDetails',
    'IssueLinkProjectDetails1',
    'IssueLinkProjectDetails2',
    'IssueLinkProjectDetails3',
    'IssueLinkScope',
    'IssueLinkScope1',
    'IssueLinkScope2',
    'IssueLinkScope3',
    'IssueLinkStatusDetails',
    'IssueLinkStatusDetails1',
    'IssueLinkType',
    'IssueLinkTypes',
    'IssueLinkUnnamedModel',
    'IssueLinkUnnamedModel1',
    'IssueLinkUnnamedModel10',
    'IssueLinkUnnamedModel11',
    'IssueLinkUnnamedModel12',
    'IssueLinkUnnamedModel13',
    'IssueLinkUnnamedModel14',
    'IssueLinkUnnamedModel15',
    'IssueLinkUnnamedModel16',
    'IssueLinkUnnamedModel17',
    'IssueLinkUnnamedModel18',
    'IssueLinkUnnamedModel19',
    'IssueLinkUnnamedModel2',
    'IssueLinkUnnamedModel20',
    'IssueLinkUnnamedModel21',
    'IssueLinkUnnamedModel22',
    'IssueLinkUnnamedModel23',
    'IssueLinkUnnamedModel24',
    'IssueLinkUnnamedModel25',
    'IssueLinkUnnamedModel26',
    'IssueLinkUnnamedModel27',
    'IssueLinkUnnamedModel28',
    'IssueLinkUnnamedModel29',
    'IssueLinkUnnamedModel3',
    'IssueLinkUnnamedModel30',
    'IssueLinkUnnamedModel31',
    'IssueLinkUnnamedModel32',
    'IssueLinkUnnamedModel33',
    'IssueLinkUnnamedModel34',
    'IssueLinkUnnamedModel35',
    'IssueLinkUnnamedModel36',
    'IssueLinkUnnamedModel4',
    'IssueLinkUnnamedModel5',
    'IssueLinkUnnamedModel6',
    'IssueLinkUnnamedModel7',
    'IssueLinkUnnamedModel8',
    'IssueLinkUnnamedModel9',
    'IssueLinkUserDetails',
    'IssueLinkUserDetails1',
    'IssueList',
    'IssueMatches',
    'IssueMatchesForJQL',
    'IssuePickerSuggestions',
    'IssuePickerSuggestionsIssueType',
    'IssueSecurityLevelMember',
    'IssueSecurityLevelMemberUnnamedModel',
    'IssueSecuritySchemeToProjectMapping',
    'IssueTransition',
    'IssueTransitionStatus',
    'IssueTypeCreateBean',
    'IssueTypeCreateBeanType',
    'IssueTypeDetails',
    'IssueTypeDetailsProjectTypeKey',
    'IssueTypeDetailsType',
    'IssueTypeDetailsUnnamedModel',
    'IssueTypeDetailsUnnamedModel1',
    'IssueTypeDetailsUnnamedModel2',
    'IssueTypeDetailsUnnamedModel3',
    'IssueTypeIds',
    'IssueTypeIdsToRemove',
    'IssueTypeInfo',
    'IssueTypeIssueCreateMetadata',
    'IssueTypeIssueCreateMetadataProjectDetails',
    'IssueTypeIssueCreateMetadataScope',
    'IssueTypeIssueCreateMetadataUnnamedModel',
    'IssueTypeIssueCreateMetadataUnnamedModel1',
    'IssueTypeIssueCreateMetadataUnnamedModel2',
    'IssueTypeIssueCreateMetadataUnnamedModel3',
    'IssueTypeProjectCreatePayload',
    'IssueTypeScheme',
    'IssueTypeSchemeDetails',
    'IssueTypeSchemeID',
    'IssueTypeSchemeMapping',
    'IssueTypeSchemePayload',
    'IssueTypeSchemeProjectAssociation',
    'IssueTypeSchemeProjects',
    'IssueTypeSchemeProjectsUnnamedModel',
    'IssueTypeSchemeUpdateDetails',
    'IssueTypeScreenScheme',
    'IssueTypeScreenSchemeDetails',
    'IssueTypeScreenSchemeId',
    'IssueTypeScreenSchemeItem',
    'IssueTypeScreenSchemeMapping',
    'IssueTypeScreenSchemeMappingDetails',
    'IssueTypeScreenSchemePayload',
    'IssueTypeScreenSchemeProjectAssociation',
    'IssueTypeScreenSchemeUpdateDetails',
    'IssueTypeScreenSchemesProjects',
    'IssueTypeScreenSchemesProjectsUnnamedModel',
    'IssueTypeToContextMapping',
    'IssueTypeUpdateBean',
    'IssueTypeWithStatus',
    'IssueTypeWorkflowMapping',
    'IssueTypesWorkflowMapping',
    'IssueUpdateDetails',
    'IssueUpdateDetailsHistoryMetadata',
    'IssueUpdateDetailsProjectDetails',
    'IssueUpdateDetailsScope',
    'IssueUpdateDetailsUnnamedModel',
    'IssueUpdateDetailsUnnamedModel1',
    'IssueUpdateDetailsUnnamedModel10',
    'IssueUpdateDetailsUnnamedModel2',
    'IssueUpdateDetailsUnnamedModel3',
    'IssueUpdateDetailsUnnamedModel4',
    'IssueUpdateDetailsUnnamedModel5',
    'IssueUpdateDetailsUnnamedModel6',
    'IssueUpdateDetailsUnnamedModel7',
    'IssueUpdateDetailsUnnamedModel8',
    'IssueUpdateDetailsUnnamedModel9',
    'IssueUpdateMetadata',
    'IssuesAndJQLQueries',
    'IssuesJqlMetaDataBean',
    'IssuesMetaBean',
    'IssuesUpdateBean',
    'JExpEvaluateIssuesJqlMetaDataBean',
    'JExpEvaluateIssuesMetaBean',
    'JExpEvaluateJiraExpressionResultBean',
    'JExpEvaluateJiraExpressionResultBeanJiraExpressionsComplexityBean',
    'JExpEvaluateJiraExpressionResultBeanUnnamedModel',
    'JExpEvaluateJiraExpressionResultBeanUnnamedModel1',
    'JExpEvaluateJiraExpressionResultBeanUnnamedModel2',
    'JExpEvaluateJiraExpressionResultBeanUnnamedModel3',
    'JExpEvaluateJiraExpressionResultBeanUnnamedModel4',
    'JExpEvaluateJiraExpressionResultBeanUnnamedModel5',
    'JExpEvaluateJiraExpressionResultBeanUnnamedModel6',
    'JExpEvaluateMetaDataBean',
    'JQLCountRequestBean',
    'JQLCountResultsBean',
    'JQLPersonalDataMigrationRequest',
    'JQLQueryWithUnknownUsers',
    'JQLReferenceData',
    'JexpEvaluateCtxIssues',
    'JexpEvaluateCtxJqlIssues',
    'JexpIssues',
    'JexpJqlIssues',
    'JiraCascadingSelectField',
    'JiraColorField',
    'JiraColorInput',
    'JiraComponentField',
    'JiraDateField',
    'JiraDateInput',
    'JiraDateTimeField',
    'JiraDateTimeInput',
    'JiraDurationField',
    'JiraExpressionAnalysis',
    'JiraExpressionComplexity',
    'JiraExpressionEvalContextBean',
    'JiraExpressionEvalRequestBean',
    'JiraExpressionEvalRequestBeanUnnamedModel',
    'JiraExpressionEvalRequestBeanUnnamedModel1',
    'JiraExpressionEvalRequestBeanUnnamedModel2',
    'JiraExpressionEvalRequestBeanUnnamedModel3',
    'JiraExpressionEvalRequestBeanUnnamedModel4',
    'JiraExpressionEvalRequestBeanValidation',
    'JiraExpressionEvaluateContextBean',
    'JiraExpressionEvaluateRequestBean',
    'JiraExpressionEvaluateRequestBeanUnnamedModel',
    'JiraExpressionEvaluateRequestBeanUnnamedModel1',
    'JiraExpressionEvaluateRequestBeanUnnamedModel2',
    'JiraExpressionEvaluateRequestBeanUnnamedModel3',
    'JiraExpressionEvaluateRequestBeanUnnamedModel4',
    'JiraExpressionEvaluationMetaDataBean',
    'JiraExpressionForAnalysis',
    'JiraExpressionResult',
    'JiraExpressionResultUnnamedModel',
    'JiraExpressionResultUnnamedModel1',
    'JiraExpressionResultUnnamedModel2',
    'JiraExpressionResultUnnamedModel3',
    'JiraExpressionResultUnnamedModel4',
    'JiraExpressionResultUnnamedModel5',
    'JiraExpressionResultUnnamedModel6',
    'JiraExpressionValidationError',
    'JiraExpressionValidationErrorType',
    'JiraExpressionsAnalysis',
    'JiraExpressionsComplexityBean',
    'JiraExpressionsComplexityValueBean',
    'JiraGroupInput',
    'JiraIssueFields',
    'JiraIssueTypeField',
    'JiraLabelPropertiesInputJackson1',
    'JiraLabelPropertiesInputJackson1Color',
    'JiraLabelsField',
    'JiraLabelsInput',
    'JiraMultiSelectComponentField',
    'JiraMultipleGroupPickerField',
    'JiraMultipleSelectField',
    'JiraMultipleSelectUserPickerField',
    'JiraMultipleVersionPickerField',
    'JiraNumberField',
    'JiraPriorityField',
    'JiraRichTextField',
    'JiraRichTextInput',
    'JiraSelectedOptionField',
    'JiraSingleGroupPickerField',
    'JiraSingleLineTextField',
    'JiraSingleSelectField',
    'JiraSingleSelectUserPickerField',
    'JiraSingleVersionPickerField',
    'JiraStatus',
    'JiraStatusInput',
    'JiraTimeTrackingField',
    'JiraUrlField',
    'JiraUserField',
    'JiraVersionField',
    'JiraWorkflow',
    'JiraWorkflowPreviewStatus',
    'JiraWorkflowStatus',
    'JqlFunctionPrecomputationBean',
    'JqlFunctionPrecomputationGetByIdRequest',
    'JqlFunctionPrecomputationGetByIdResponse',
    'JqlFunctionPrecomputationUpdateBean',
    'JqlFunctionPrecomputationUpdateErrorResponse',
    'JqlFunctionPrecomputationUpdateRequestBean',
    'JqlFunctionPrecomputationUpdateResponse',
    'JqlQueriesToParse',
    'JqlQueriesToSanitize',
    'JqlQuery',
    'JqlQueryClauseTimePredicate',
    'JqlQueryClauseTimePredicateOperator',
    'JqlQueryField',
    'JqlQueryFieldEntityProperty',
    'JqlQueryFieldEntityPropertyType',
    'JqlQueryOrderByClause',
    'JqlQueryOrderByClauseElement',
    'JqlQueryOrderByClauseElementDirection',
    'JqlQueryToSanitize',
    'JsonContextVariable',
    'JsonNode',
    'JsonNodeNumberType',
    'JsonTypeBean',
    'KeywordOperand',
    'KeywordOperandKeyword',
    'License',
    'LicenseMetric',
    'LicensedApplication',
    'LicensedApplicationPlan',
    'LinkGroup',
    'LinkIssueRequestJsonBean',
    'LinkedIssue',
    'LinkedIssueIssueTypeDetails',
    'LinkedIssueProjectDetails',
    'LinkedIssueProjectDetails1',
    'LinkedIssueScope',
    'LinkedIssueScope1',
    'LinkedIssueStatusDetails',
    'LinkedIssueUnnamedModel',
    'LinkedIssueUnnamedModel1',
    'LinkedIssueUnnamedModel10',
    'LinkedIssueUnnamedModel11',
    'LinkedIssueUnnamedModel12',
    'LinkedIssueUnnamedModel13',
    'LinkedIssueUnnamedModel14',
    'LinkedIssueUnnamedModel15',
    'LinkedIssueUnnamedModel16',
    'LinkedIssueUnnamedModel2',
    'LinkedIssueUnnamedModel3',
    'LinkedIssueUnnamedModel4',
    'LinkedIssueUnnamedModel5',
    'LinkedIssueUnnamedModel6',
    'LinkedIssueUnnamedModel7',
    'LinkedIssueUnnamedModel8',
    'LinkedIssueUnnamedModel9',
    'LinkedIssueUserDetails',
    'ListOperand',
    'Locale',
    'MappingsByIssueTypeOverride',
    'MappingsByWorkflow',
    'MinimalFieldSchemeToFieldsPartialFailure',
    'MinimalFieldSchemeToFieldsResponse',
    'MoveFieldBean',
    'MoveFieldBeanPosition',
    'MultiIssueEntityProperties',
    'MultipartFile',
    'MultipleCustomFieldValuesUpdate',
    'MultipleCustomFieldValuesUpdateDetails',
    'NestedResponse',
    'NewUserDetails',
    'NonWorkingDay',
    'Notification',
    'NotificationEvent',
    'NotificationEventUnnamedModel',
    'NotificationRecipients',
    'NotificationRecipientsRestrictions',
    'NotificationScheme',
    'NotificationSchemeAndProjectMappingJsonBean',
    'NotificationSchemeEvent',
    'NotificationSchemeEventDetails',
    'NotificationSchemeEventDetailsUnnamedModel',
    'NotificationSchemeEventIDPayload',
    'NotificationSchemeEventPayload',
    'NotificationSchemeEventTypeId',
    'NotificationSchemeId',
    'NotificationSchemeNotificationDetails',
    'NotificationSchemeNotificationDetailsPayload',
    'NotificationSchemePayload',
    'NotificationSchemeProjectDetails',
    'NotificationSchemeScope',
    'NotificationSchemeUnnamedModel',
    'NotificationSchemeUnnamedModel1',
    'NotificationSchemeUnnamedModel2',
    'NotificationSchemeUnnamedModel3',
    'NotificationUnnamedModel',
    'NotificationUnnamedModel1',
    'OldToNewSecurityLevelMappingsBean',
    'OperationMessage',
    'Operations',
    'OrderOfCustomFieldOptions',
    'OrderOfCustomFieldOptionsPosition',
    'OrderOfIssueTypes',
    'PageBean2ComponentJsonBean',
    'PageBean2FieldAssociationSchemeFieldSearchResult',
    'PageBean2FieldAssociationSchemeProjectSearchResult',
    'PageBean2GetFieldAssociationSchemeResponse',
    'PageBean2GetProjectsWithFieldSchemesResponse',
    'PageBean2JqlFunctionPrecomputationBean',
    'PageBean2ProjectFieldBean',
    'PageBeanBulkContextualConfiguration',
    'PageBeanChangelog',
    'PageBeanComment',
    'PageBeanComponentWithIssueCount',
    'PageBeanContext',
    'PageBeanContextForProjectAndIssueType',
    'PageBeanContextualConfiguration',
    'PageBeanCustomFieldContext',
    'PageBeanCustomFieldContextDefaultValue',
    'PageBeanCustomFieldContextOption',
    'PageBeanCustomFieldContextProjectMapping',
    'PageBeanDashboard',
    'PageBeanField',
    'PageBeanFieldConfigurationDetails',
    'PageBeanFieldConfigurationIssueTypeItem',
    'PageBeanFieldConfigurationItem',
    'PageBeanFieldConfigurationScheme',
    'PageBeanFieldConfigurationSchemeProjects',
    'PageBeanFieldProjectAssociation',
    'PageBeanFilterDetails',
    'PageBeanGroupDetails',
    'PageBeanIssueFieldOption',
    'PageBeanIssueSecurityLevelMember',
    'PageBeanIssueSecuritySchemeToProjectMapping',
    'PageBeanIssueTypeScheme',
    'PageBeanIssueTypeSchemeMapping',
    'PageBeanIssueTypeSchemeProjects',
    'PageBeanIssueTypeScreenScheme',
    'PageBeanIssueTypeScreenSchemeItem',
    'PageBeanIssueTypeScreenSchemesProjects',
    'PageBeanIssueTypeToContextMapping',
    'PageBeanNotificationScheme',
    'PageBeanNotificationSchemeAndProjectMappingJsonBean',
    'PageBeanPriority',
    'PageBeanPrioritySchemeWithPaginatedPrioritiesAndProjects',
    'PageBeanPriorityWithSequence',
    'PageBeanProject',
    'PageBeanProjectDetails',
    'PageBeanResolutionJsonBean',
    'PageBeanScreen',
    'PageBeanScreenScheme',
    'PageBeanScreenWithTab',
    'PageBeanSecurityLevel',
    'PageBeanSecurityLevelMember',
    'PageBeanSecuritySchemeWithProjects',
    'PageBeanString',
    'PageBeanUiModificationDetails',
    'PageBeanUser',
    'PageBeanUserDetails',
    'PageBeanUserKey',
    'PageBeanVersion',
    'PageBeanWebhook',
    'PageBeanWorkflow',
    'PageBeanWorkflowScheme',
    'PageBeanWorkflowTransitionRules',
    'PageOfChangelogs',
    'PageOfComments',
    'PageOfCreateMetaIssueTypeWithField',
    'PageOfCreateMetaIssueTypes',
    'PageOfDashboards',
    'PageOfStatuses',
    'PageOfWorklogs',
    'PageWithCursorGetPlanResponseForPage',
    'PageWithCursorGetTeamResponseForPage',
    'PagedListUserDetailsApplicationUser',
    'ParameterRemovalDetails',
    'ParsedJqlQueries',
    'ParsedJqlQuery',
    'ParsedJqlQueryUnnamedModel',
    'PermissionDetails',
    'PermissionGrant',
    'PermissionGrantDTO',
    'PermissionGrantUnnamedModel',
    'PermissionGrants',
    'PermissionHolder',
    'PermissionPayloadDTO',
    'PermissionScheme',
    'PermissionSchemeProjectDetails',
    'PermissionSchemeScope',
    'PermissionSchemeUnnamedModel',
    'PermissionSchemeUnnamedModel1',
    'PermissionSchemeUnnamedModel2',
    'PermissionSchemeUnnamedModel3',
    'PermissionSchemes',
    'Permissions',
    'PermissionsKeysBean',
    'PermittedProjects',
    'PreviewConditionGroupConfiguration',
    'PreviewRuleConfiguration',
    'PreviewTrigger',
    'Priority',
    'PriorityId',
    'PriorityMapping',
    'PrioritySchemeChangesWithoutMappings',
    'PrioritySchemeId',
    'PrioritySchemeIdUnnamedModel',
    'PrioritySchemeIdUnnamedModel1',
    'PrioritySchemeWithPaginatedPrioritiesAndProjects',
    'PrioritySchemeWithPaginatedPrioritiesAndProjectsUnnamedModel',
    'PrioritySchemeWithPaginatedPrioritiesAndProjectsUnnamedModel1',
    'PriorityWithSequence',
    'Project',
    'ProjectAndIssueTypePair',
    'ProjectArchetype',
    'ProjectArchetypeRealType',
    'ProjectAvatars',
    'ProjectCategory',
    'ProjectComponent',
    'ProjectComponentAssigneeType',
    'ProjectComponentUnnamedModel',
    'ProjectComponentUnnamedModel1',
    'ProjectComponentUnnamedModel2',
    'ProjectComponentUnnamedModel3',
    'ProjectComponentUnnamedModel4',
    'ProjectComponentUnnamedModel5',
    'ProjectComponentUnnamedModel6',
    'ProjectComponentUnnamedModel7',
    'ProjectComponentUnnamedModel8',
    'ProjectComponentUser',
    'ProjectComponentUser1',
    'ProjectComponentUser2',
    'ProjectCreateResourceIdentifier',
    'ProjectCreateResourceIdentifierType',
    'ProjectCustomTemplateCreateRequestDTO',
    'ProjectDataPolicies',
    'ProjectDataPolicy',
    'ProjectDetails',
    'ProjectEmailAddress',
    'ProjectFeature',
    'ProjectFeatureState',
    'ProjectFieldBean',
    'ProjectId',
    'ProjectIdentifierBean',
    'ProjectIdentifiers',
    'ProjectIds',
    'ProjectInsight',
    'ProjectIssueCreateMetadata',
    'ProjectIssueCreateMetadataUnnamedModel',
    'ProjectIssueSecurityLevels',
    'ProjectIssueTypeHierarchy',
    'ProjectIssueTypeMapping',
    'ProjectIssueTypeMappings',
    'ProjectIssueTypeQueryContext',
    'ProjectIssueTypesHierarchyLevel',
    'ProjectLandingPageInfo',
    'ProjectPayload',
    'ProjectPayloadProjectTypeKey',
    'ProjectPermissions',
    'ProjectPinAction',
    'ProjectPinActionAction',
    'ProjectRole',
    'ProjectRoleActorsUpdateBean',
    'ProjectRoleDetails',
    'ProjectRoleDetailsProjectDetails',
    'ProjectRoleDetailsScope',
    'ProjectRoleDetailsType',
    'ProjectRoleDetailsUnnamedModel',
    'ProjectRoleDetailsUnnamedModel1',
    'ProjectRoleDetailsUnnamedModel2',
    'ProjectRoleDetailsUnnamedModel3',
    'ProjectRoleGroup',
    'ProjectRoleUser',
    'ProjectScopeBean',
    'ProjectTemplateKey',
    'ProjectTemplateModel',
    'ProjectTemplateModelType',
    'ProjectType',
    'ProjectUsage',
    'ProjectUsagePage',
    'ProjectWithDataPolicy',
    'ProjectWithDataPolicyUnnamedModel',
    'PropertyKey',
    'PropertyKeys',
    'PublishDraftWorkflowScheme',
    'PublishedWorkflowId',
    'QuickFilterPayload',
    'RedactionJobStatusResponse',
    'RedactionJobStatusResponseJobStatus',
    'RedactionPosition',
    'RegisteredWebhook',
    'RemoteIssueLink',
    'RemoteIssueLinkIdentifies',
    'RemoteIssueLinkRequest',
    'RemoteIssueLinkRequestRemoteObject',
    'RemoteIssueLinkRequestStatus',
    'RemoteIssueLinkRequestUnnamedModel',
    'RemoteIssueLinkRequestUnnamedModel1',
    'RemoteIssueLinkRequestUnnamedModel2',
    'RemoteIssueLinkRequestUnnamedModel3',
    'RemoteIssueLinkRequestUnnamedModel4',
    'RemoteIssueLinkUnnamedModel',
    'RemoteIssueLinkUnnamedModel1',
    'RemoteIssueLinkUnnamedModel2',
    'RemoteIssueLinkUnnamedModel3',
    'RemoteIssueLinkUnnamedModel4',
    'RemoteObject',
    'RemoveFieldAssociationsRequestItem',
    'RemoveFieldParametersResult',
    'RemoveFieldParametersResultError',
    'RemoveOptionFromIssuesResult',
    'ReorderIssuePriorities',
    'ReorderIssueResolutionsRequest',
    'RequiredMappingByIssueType',
    'RequiredMappingByWorkflows',
    'Resolution',
    'ResolutionId',
    'ResolutionJsonBean',
    'Resource',
    'RestrictedPermission',
    'RoleActor',
    'RoleActorType',
    'RoleActorUnnamedModel',
    'RoleActorUnnamedModel1',
    'RolePayload',
    'RolePayloadType',
    'RolesCapabilityPayload',
    'RuleConfiguration',
    'RulePayload',
    'SanitizedJqlQueries',
    'SanitizedJqlQuery',
    'SanitizedJqlQueryUnnamedModel',
    'SaveProjectTemplateRequest',
    'SaveTemplateRequest',
    'SaveTemplateResponse',
    'Scope',
    'ScopePayload',
    'Screen',
    'ScreenDetails',
    'ScreenProjectDetails',
    'ScreenScheme',
    'ScreenSchemeDetails',
    'ScreenSchemeDetailsUnnamedModel',
    'ScreenSchemeId',
    'ScreenSchemeUnnamedModel',
    'ScreenSchemeUnnamedModel1',
    'ScreenScope',
    'ScreenTypes',
    'ScreenUnnamedModel',
    'ScreenUnnamedModel1',
    'ScreenUnnamedModel2',
    'ScreenUnnamedModel3',
    'ScreenWithTab',
    'ScreenWithTabProjectDetails',
    'ScreenWithTabScope',
    'ScreenWithTabUnnamedModel',
    'ScreenWithTabUnnamedModel1',
    'ScreenWithTabUnnamedModel2',
    'ScreenWithTabUnnamedModel3',
    'ScreenWithTabUnnamedModel4',
    'ScreenableField',
    'ScreenableTab',
    'SearchAndReconcileRequestBean',
    'SearchAndReconcileResults',
    'SearchAutoCompleteFilter',
    'SearchRequestBean',
    'SearchResultFieldParameters',
    'SearchResultWorkTypeParameters',
    'SearchResults',
    'SearchWarning',
    'SearchWarningLimitDetails',
    'SearchWarningUnnamedModel',
    'SecurityLevel',
    'SecurityLevelMember',
    'SecurityLevelMemberPayload',
    'SecurityLevelMemberPayloadType',
    'SecurityLevelMemberUnnamedModel',
    'SecurityLevelPayload',
    'SecurityScheme',
    'SecuritySchemeId',
    'SecuritySchemeLevelBean',
    'SecuritySchemeLevelMemberBean',
    'SecuritySchemeMembersRequest',
    'SecuritySchemePayload',
    'SecuritySchemeWithProjects',
    'SecuritySchemes',
    'ServerInformation',
    'ServiceRegistry',
    'ServiceRegistryTier',
    'SetDefaultLevelsRequest',
    'SetDefaultPriorityRequest',
    'SetDefaultResolutionRequest',
    'SharePermission',
    'SharePermissionAssigneeType',
    'SharePermissionInputBean',
    'SharePermissionInputBeanType',
    'SharePermissionProjectDetails',
    'SharePermissionScope',
    'SharePermissionStyle',
    'SharePermissionType',
    'SharePermissionUnnamedModel',
    'SharePermissionUnnamedModel1',
    'SharePermissionUnnamedModel10',
    'SharePermissionUnnamedModel11',
    'SharePermissionUnnamedModel12',
    'SharePermissionUnnamedModel13',
    'SharePermissionUnnamedModel14',
    'SharePermissionUnnamedModel15',
    'SharePermissionUnnamedModel16',
    'SharePermissionUnnamedModel17',
    'SharePermissionUnnamedModel18',
    'SharePermissionUnnamedModel19',
    'SharePermissionUnnamedModel2',
    'SharePermissionUnnamedModel20',
    'SharePermissionUnnamedModel21',
    'SharePermissionUnnamedModel22',
    'SharePermissionUnnamedModel23',
    'SharePermissionUnnamedModel3',
    'SharePermissionUnnamedModel4',
    'SharePermissionUnnamedModel5',
    'SharePermissionUnnamedModel6',
    'SharePermissionUnnamedModel7',
    'SharePermissionUnnamedModel8',
    'SharePermissionUnnamedModel9',
    'SharePermissionUser',
    'SharePermissionUser1',
    'SharePermissionUser2',
    'SimpleApplicationPropertyBean',
    'SimpleErrorCollection',
    'SimpleLink',
    'SimpleListWrapperApplicationRole',
    'SimpleListWrapperGroupName',
    'SimplifiedHierarchyLevel',
    'SimplifiedIssueTransition',
    'SimplifiedIssueTransitionUnnamedModel',
    'SingleRedactionRequest',
    'SingleRedactionResponse',
    'Status',
    'StatusCategory',
    'StatusCreate',
    'StatusCreateRequest',
    'StatusDetails',
    'StatusLayoutUpdate',
    'StatusMapping',
    'StatusMappingDTO',
    'StatusMetadata',
    'StatusMigration',
    'StatusPayload',
    'StatusPayloadStatusCategory',
    'StatusProjectIssueTypeUsage',
    'StatusProjectIssueTypeUsageDTO',
    'StatusProjectIssueTypeUsagePage',
    'StatusProjectUsage',
    'StatusProjectUsageDTO',
    'StatusProjectUsagePage',
    'StatusScope',
    'StatusUpdate',
    'StatusUpdateRequest',
    'StatusWorkflowUsageDTO',
    'StatusWorkflowUsagePage',
    'StatusWorkflowUsageWorkflow',
    'StatusesPerWorkflow',
    'SubmittedBulkOperation',
    'SuccessOrErrorResults',
    'SuggestedIssue',
    'SuggestedMappingsForPrioritiesRequestBean',
    'SuggestedMappingsForProjectsRequestBean',
    'SuggestedMappingsRequestBean',
    'SuggestedMappingsRequestBeanUnnamedModel',
    'SuggestedMappingsRequestBeanUnnamedModel1',
    'SwimlanePayload',
    'SwimlanesPayload',
    'SwimlanesPayloadSwimlaneStrategy',
    'SystemAvatars',
    'TaskProgress',
    'TaskProgressBeanJsonNode',
    'TaskProgressBeanObject',
    'TaskProgressBeanRemoveOptionFromIssuesResult',
    'TaskProgressBeanRemoveOptionFromIssuesResultUnnamedModel',
    'TaskProgressBeanRemoveOptionFromIssuesResultUnnamedModel1',
    'TimeTrackingConfiguration',
    'TimeTrackingDetails',
    'TimeTrackingProvider',
    'ToLayoutPayload',
    'Transition',
    'TransitionLink',
    'TransitionPayload',
    'TransitionPayloadType',
    'TransitionPreview',
    'TransitionScreenDetails',
    'TransitionUpdateDTO',
    'Transitions',
    'UiModificationContextDetails',
    'UiModificationContextDetailsViewType',
    'UiModificationDetails',
    'UiModificationIdentifiers',
    'UnnamedModel',
    'UnrestrictedUserEmail',
    'UpdateCustomFieldDetails',
    'UpdateDefaultProjectClassificationBean',
    'UpdateDefaultScreenScheme',
    'UpdateFieldAssociationSchemeLinksBean',
    'UpdateFieldAssociationSchemeRequest',
    'UpdateFieldAssociationSchemeResponse',
    'UpdateFieldAssociationsRequestItem',
    'UpdateFieldConfigurationSchemeDetails',
    'UpdateFieldSchemeParametersPartialFailure',
    'UpdateFieldSchemeParametersRequest',
    'UpdateFieldSchemeParametersResponse',
    'UpdateIssueSecurityLevelDetails',
    'UpdateIssueSecuritySchemeRequestBean',
    'UpdateNotificationSchemeDetails',
    'UpdatePrioritiesInSchemeRequestBean',
    'UpdatePriorityDetails',
    'UpdatePrioritySchemeRequestBean',
    'UpdatePrioritySchemeRequestBeanUnnamedModel',
    'UpdatePrioritySchemeRequestBeanUnnamedModel1',
    'UpdatePrioritySchemeRequestBeanUnnamedModel2',
    'UpdatePrioritySchemeRequestBeanUnnamedModel3',
    'UpdatePrioritySchemeRequestBeanUnnamedModel4',
    'UpdatePrioritySchemeRequestBeanUnnamedModel5',
    'UpdatePrioritySchemeRequestBeanUnnamedModel6',
    'UpdatePrioritySchemeResponseBean',
    'UpdatePrioritySchemeResponseBeanTaskProgressBeanJsonNode',
    'UpdatePrioritySchemeResponseBeanUnnamedModel',
    'UpdatePrioritySchemeResponseBeanUnnamedModel1',
    'UpdateProjectDetails',
    'UpdateProjectsInSchemeRequestBean',
    'UpdateResolutionDetails',
    'UpdateScreenDetails',
    'UpdateScreenSchemeDetails',
    'UpdateScreenSchemeDetailsUnnamedModel',
    'UpdateScreenTypes',
    'UpdateUiModificationDetails',
    'UpdateUserToGroupBean',
    'UpdatedProjectCategory',
    'User',
    'UserBean',
    'UserBeanAvatarUrls',
    'UserColumnRequestBody',
    'UserContextVariable',
    'UserDetails',
    'UserFilter',
    'UserKey',
    'UserList',
    'UserMigrationBean',
    'UserPermission',
    'UserPermissionType',
    'UserPickerUser',
    'ValidationOptionsForCreate',
    'ValidationOptionsForCreateEnum',
    'ValidationOptionsForUpdate',
    'ValueOperand',
    'Version',
    'VersionApprover',
    'VersionIssueCounts',
    'VersionIssuesStatus',
    'VersionMoveBean',
    'VersionRelatedWork',
    'VersionUnnamedModel',
    'VersionUnresolvedIssuesCount',
    'VersionUsageInCustomField',
    'Visibility',
    'Votes',
    'WarningCollection',
    'Watchers',
    'Webhook',
    'WebhookDetails',
    'WebhookEnum',
    'WebhookRegistrationDetails',
    'WebhooksExpirationDate',
    'WorkTypeParameters',
    'Workflow',
    'WorkflowAssociationStatusMapping',
    'WorkflowCapabilities',
    'WorkflowCapabilitiesEnum',
    'WorkflowCapabilityPayload',
    'WorkflowCompoundCondition',
    'WorkflowCompoundConditionOperator',
    'WorkflowCreate',
    'WorkflowCreateRequest',
    'WorkflowCreateResponse',
    'WorkflowCreateValidateRequest',
    'WorkflowDocumentDTO',
    'WorkflowDocumentStatusDTO',
    'WorkflowDocumentVersionBean',
    'WorkflowElementReference',
    'WorkflowHistoryItemDTO',
    'WorkflowHistoryListRequest',
    'WorkflowHistoryListResponseDTO',
    'WorkflowHistoryReadRequest',
    'WorkflowHistoryReadResponseDTO',
    'WorkflowId',
    'WorkflowLayout',
    'WorkflowMetadataAndIssueTypeRestModel',
    'WorkflowMetadataRestModel',
    'WorkflowOperations',
    'WorkflowPayload',
    'WorkflowPreview',
    'WorkflowPreviewLayout',
    'WorkflowPreviewRequest',
    'WorkflowPreviewResponse',
    'WorkflowPreviewScope',
    'WorkflowPreviewStatus',
    'WorkflowProjectIdScope',
    'WorkflowProjectIssueTypeUsage',
    'WorkflowProjectIssueTypeUsageDTO',
    'WorkflowProjectIssueTypeUsagePage',
    'WorkflowProjectUsageDTO',
    'WorkflowReadRequest',
    'WorkflowReadResponse',
    'WorkflowReferenceStatus',
    'WorkflowRuleConfiguration',
    'WorkflowRules',
    'WorkflowRulesSearch',
    'WorkflowRulesSearchDetails',
    'WorkflowScheme',
    'WorkflowSchemeAssociation',
    'WorkflowSchemeAssociations',
    'WorkflowSchemeAssociationsUnnamedModel',
    'WorkflowSchemeAssociationsUnnamedModel1',
    'WorkflowSchemeAssociationsUnnamedModel2',
    'WorkflowSchemeAssociationsUnnamedModel3',
    'WorkflowSchemeAssociationsUser',
    'WorkflowSchemeAssociationsWorkflowScheme',
    'WorkflowSchemeIdName',
    'WorkflowSchemePayload',
    'WorkflowSchemeProjectAssociation',
    'WorkflowSchemeProjectSwitchBean',
    'WorkflowSchemeProjectUsageDTO',
    'WorkflowSchemeReadRequest',
    'WorkflowSchemeReadResponse',
    'WorkflowSchemeUnnamedModel',
    'WorkflowSchemeUnnamedModel1',
    'WorkflowSchemeUnnamedModel2',
    'WorkflowSchemeUpdateRequest',
    'WorkflowSchemeUpdateRequiredMappingsRequest',
    'WorkflowSchemeUpdateRequiredMappingsResponse',
    'WorkflowSchemeUsage',
    'WorkflowSchemeUsageDTO',
    'WorkflowSchemeUsagePage',
    'WorkflowSchemeUser',
    'WorkflowScope',
    'WorkflowSearchResponse',
    'WorkflowSimpleCondition',
    'WorkflowStatus',
    'WorkflowStatusLayout',
    'WorkflowStatusLayoutPayload',
    'WorkflowStatusPayload',
    'WorkflowStatusUpdate',
    'WorkflowTransition',
    'WorkflowTransitionLinks',
    'WorkflowTransitionProperty',
    'WorkflowTransitionRule',
    'WorkflowTransitionRules',
    'WorkflowTransitionRulesDetails',
    'WorkflowTransitionRulesUpdate',
    'WorkflowTransitionRulesUpdateErrorDetails',
    'WorkflowTransitionRulesUpdateErrors',
    'WorkflowTransitions',
    'WorkflowTransitionsType',
    'WorkflowTrigger',
    'WorkflowUpdate',
    'WorkflowUpdateRequest',
    'WorkflowUpdateResponse',
    'WorkflowUpdateValidateRequestBean',
    'WorkflowValidationError',
    'WorkflowValidationErrorList',
    'WorkflowValidationErrorType',
    'WorkflowsWithTransitionRulesDetails',
    'WorkingDaysConfig',
    'Worklog',
    'WorklogCompositeKey',
    'WorklogIdsRequestBean',
    'WorklogKeyResult',
    'WorklogUnnamedModel',
    'WorklogUnnamedModel1',
    'WorklogUnnamedModel2',
    'WorklogUnnamedModel3',
    'WorklogUnnamedModel4',
    'WorklogUserDetails',
    'WorklogUserDetails1',
    'WorklogsMoveRequestBean',
    'WorkspaceDataPolicy',
    'getForgeAppPropertyKeysResponseUnnamedModel',
    'getForgeAppPropertyKeysResponseUnnamedModel1',
    'getForgeAppPropertyResponseUnnamedModel',
    'targetToSourcesMapping',
)


def _html_val(v):
    if v is None:
        return '<em style="color:#aaa">None</em>'
    if isinstance(v, list):
        n = len(v)
        preview = ', '.join((str(x) for x in v[:3]))
        suffix = f', … ({n})' if n > 3 else ''
        return f'[{preview}{suffix}]'
    if hasattr(v, '_repr_html_'):
        return v._repr_html_()
    return str(v)


class _HtmlReprMixin:
    """Mixin that renders a Pydantic model as an HTML table in Jupyter notebooks."""

    def _repr_html_(self) -> str:
        fields = getattr(self.__class__, 'model_fields', {})
        rows = []
        for i, name in enumerate(fields):
            val = getattr(self, name, None)
            bg = ' style="background:#f8f8f8"' if i % 2 == 0 else ''
            rows.append(
                f'<tr{bg}><td style="font-weight:bold;padding:2px 8px;white-space:nowrap;color:#444">{name}</td><td style="padding:2px 8px">{_html_val(val)}</td></tr>'
            )
        class_name = type(self).__name__
        inner = ''.join(rows)
        return f'<details open><summary style="font-weight:bold;cursor:pointer;font-family:monospace">{class_name}</summary><table style="border-collapse:collapse;font-size:13px;font-family:monospace">{inner}</table></details>'


class AnnouncementBannerConfigurationVisibility(_HtmlReprMixin, str, Enum):
    PUBLIC = 'public'
    PRIVATE = 'private'


class AnnouncementBannerConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    hashId: str = Field(default=None)
    isDismissible: bool = Field(default=None)
    isEnabled: bool = Field(default=None)
    message: str = Field(default=None)
    visibility: AnnouncementBannerConfigurationVisibility = Field(default=None)


class ErrorCollection(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errorMessages: list[str] = Field(default=None)
    errors: dict[str, str] = Field(default=None)
    status: int = Field(default=None)


class AnnouncementBannerConfigurationUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isDismissible: bool = Field(default=None)
    isEnabled: bool = Field(default=None)
    message: str = Field(default=None)
    visibility: str = Field(default=None)


class ConfigurationsListParameters(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldIdsOrKeys: list[str] = Field(min_length=1)


class BulkContextualConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    configuration: dict[str, Any] = Field(default=None)
    customFieldId: str
    fieldContextId: str
    id: str
    schema: dict[str, Any] = Field(default=None)


class PageBeanBulkContextualConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[BulkContextualConfiguration] = Field(default=None)


class MultipleCustomFieldValuesUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customField: str
    issueIds: list[int]
    value: dict[str, Any]


class MultipleCustomFieldValuesUpdateDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    updates: list[MultipleCustomFieldValuesUpdate] = Field(default=None)


class ContextualConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    configuration: dict[str, Any] = Field(default=None)
    fieldContextId: str
    id: str
    schema: dict[str, Any] = Field(default=None)


class PageBeanContextualConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ContextualConfiguration] = Field(default=None)


class CustomFieldConfigurations(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    configurations: list[ContextualConfiguration] = Field(min_length=1, max_length=1000)


class CustomFieldValueUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueIds: list[int]
    value: dict[str, Any]


class CustomFieldValueUpdateDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    updates: list[CustomFieldValueUpdate] = Field(default=None)


class ApplicationProperty(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    allowedValues: list[str] = Field(default=None)
    defaultValue: str = Field(default=None)
    desc: str = Field(default=None)
    example: str = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    type_: str = Field(default=None, alias='type')
    value: str = Field(default=None)


class SimpleApplicationPropertyBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    value: str = Field(default=None)


class GroupName(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    groupId: str | None = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class ApplicationRole(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultGroups: list[str] = Field(default=None)
    defaultGroupsDetails: list[GroupName] = Field(default=None)
    defined: bool = Field(default=None)
    groupDetails: list[GroupName] = Field(default=None)
    groups: list[str] = Field(default=None)
    hasUnlimitedSeats: bool = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    numberOfSeats: int = Field(default=None)
    platform: bool = Field(default=None)
    remainingSeats: int = Field(default=None)
    selectedByDefault: bool = Field(default=None)
    userCount: int = Field(default=None)
    userCountDescription: str = Field(default=None)


class AttachmentSettings(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    enabled: bool = Field(default=None)
    uploadLimit: int = Field(default=None)


class AvatarUrlsBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    field_16x16: str = Field(default=None, alias='16x16')
    field_24x24: str = Field(default=None, alias='24x24')
    field_32x32: str = Field(default=None, alias='32x32')
    field_48x48: str = Field(default=None, alias='48x48')


class AttachmentMetadataUnnamedModel(AvatarUrlsBean):
    pass


class SimpleListWrapperApplicationRole(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    callback: dict[str, Any] = Field(default=None)
    items: list[ApplicationRole] = Field(default=None)
    max_results: int = Field(default=None, alias='max-results')
    pagingCallback: dict[str, Any] = Field(default=None)
    size: int = Field(default=None)


class UnnamedModel(SimpleListWrapperApplicationRole):
    pass


class AttachmentMetadataAccountType(_HtmlReprMixin, str, Enum):
    ATLASSIAN = 'atlassian'
    APP = 'app'
    CUSTOMER = 'customer'
    UNKNOWN = 'unknown'


class SimpleListWrapperGroupName(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    callback: dict[str, Any] = Field(default=None)
    items: list[GroupName] = Field(default=None)
    max_results: int = Field(default=None, alias='max-results')
    pagingCallback: dict[str, Any] = Field(default=None)
    size: int = Field(default=None)


class AttachmentMetadataUnnamedModel1(SimpleListWrapperGroupName):
    pass


class User(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: AttachmentMetadataUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: AttachmentMetadataUnnamedModel1 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class AttachmentMetadataUnnamedModel2(User):
    pass


class AttachmentMetadata(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    author: AttachmentMetadataUnnamedModel2 = Field(default=None)
    content: str = Field(default=None)
    created: datetime = Field(default=None)
    filename: str = Field(default=None)
    id: int = Field(default=None)
    mimeType: str = Field(default=None)
    properties: dict[str, dict[str, Any]] = Field(default=None)
    self: str = Field(default=None)
    size: int = Field(default=None)
    thumbnail: str = Field(default=None)


class AttachmentArchiveItemReadable(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    index: int = Field(default=None)
    label: str = Field(default=None)
    mediaType: str = Field(default=None)
    path: str = Field(default=None)
    size: str = Field(default=None)


class AttachmentArchiveMetadataReadable(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    entries: list[AttachmentArchiveItemReadable] = Field(default=None)
    id: int = Field(default=None)
    mediaType: str = Field(default=None)
    name: str = Field(default=None)
    totalEntryCount: int = Field(default=None)


class AttachmentArchiveEntry(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    abbreviatedName: str = Field(default=None)
    entryIndex: int = Field(default=None)
    mediaType: str = Field(default=None)
    name: str = Field(default=None)
    size: int = Field(default=None)


class AttachmentArchiveImpl(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    entries: list[AttachmentArchiveEntry] = Field(default=None)
    totalEntryCount: int = Field(default=None)


class AssociatedItemBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    name: str = Field(default=None)
    parentId: str = Field(default=None)
    parentName: str = Field(default=None)
    typeName: str = Field(default=None)


class ChangedValueBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    changedFrom: str = Field(default=None)
    changedTo: str = Field(default=None)
    fieldName: str = Field(default=None)


class AuditRecordBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    associatedItems: list[AssociatedItemBean] = Field(default=None)
    authorKey: str = Field(default=None)
    category: str = Field(default=None)
    changedValues: list[ChangedValueBean] = Field(default=None)
    created: datetime = Field(default=None)
    description: str = Field(default=None)
    eventSource: str = Field(default=None)
    id: int = Field(default=None)
    objectItem: AssociatedItemBean = Field(default=None)
    remoteAddress: str = Field(default=None)
    summary: str = Field(default=None)


class AuditRecords(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    limit: int = Field(default=None)
    offset: int = Field(default=None)
    records: list[AuditRecordBean] = Field(default=None)
    total: int = Field(default=None)


class AutoEnum(_HtmlReprMixin, str, Enum):
    ISSUETYPE = 'issuetype'
    PROJECT = 'project'
    USER = 'user'
    PRIORITY = 'priority'


class Avatar(_HtmlReprMixin, BaseModel):
    fileName: str = Field(default=None)
    id: str
    isDeletable: bool = Field(default=None)
    isSelected: bool = Field(default=None)
    isSystemAvatar: bool = Field(default=None)
    owner: str = Field(default=None)
    urls: dict[str, str] = Field(default=None)


class SystemAvatars(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    system: list[Avatar] = Field(default=None)


class IssueBulkDeletePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    selectedIssueIdsOrKeys: list[str]
    sendBulkNotification: bool | None = Field(default=True)


class SubmittedBulkOperation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    taskId: str = Field(default=None)


class ErrorMessage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    message: str = Field(default=None)


class BulkOperationErrorResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errors: list[ErrorMessage] = Field(default=None)


class IssueBulkEditFieldEnum(_HtmlReprMixin, str, Enum):
    ADD = 'ADD'
    REMOVE = 'REMOVE'
    REPLACE = 'REPLACE'
    REMOVEALL = 'REMOVE_ALL'


class IssueBulkEditField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    fieldOptions: list[dict[str, Any]] = Field(default=None)
    id: str = Field(default=None)
    isRequired: bool = Field(default=None)
    multiSelectFieldOptions: list[IssueBulkEditFieldEnum] = Field(default=None)
    name: str = Field(default=None)
    searchUrl: str = Field(default=None)
    type_: str = Field(default=None, alias='type')
    unavailableMessage: str = Field(default=None)


class BulkEditGetFields(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    endingBefore: str = Field(default=None)
    fields: list[IssueBulkEditField] = Field(default=None)
    startingAfter: str = Field(default=None)


class JiraVersionField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    versionId: str = Field(default=None)


class JiraSingleVersionPickerField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    version: JiraVersionField


class JiraDateInput(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    formattedDate: str


class JiraNumberField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    value: float = Field(default=None)


class JiraIssueTypeField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str


class IssueBulkEditPayloadUnnamedModel(JiraIssueTypeField):
    pass


class JiraPriorityField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    priorityId: str


class JiraSelectedOptionField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    optionId: int = Field(default=None)


class JiraLabelPropertiesInputJackson1Color(_HtmlReprMixin, str, Enum):
    GREYLIGHTEST = 'GREY_LIGHTEST'
    GREYLIGHTER = 'GREY_LIGHTER'
    GREY = 'GREY'
    GREYDARKER = 'GREY_DARKER'
    GREYDARKEST = 'GREY_DARKEST'
    PURPLELIGHTEST = 'PURPLE_LIGHTEST'
    PURPLELIGHTER = 'PURPLE_LIGHTER'
    PURPLE = 'PURPLE'
    PURPLEDARKER = 'PURPLE_DARKER'
    PURPLEDARKEST = 'PURPLE_DARKEST'
    BLUELIGHTEST = 'BLUE_LIGHTEST'
    BLUELIGHTER = 'BLUE_LIGHTER'
    BLUE = 'BLUE'
    BLUEDARKER = 'BLUE_DARKER'
    BLUEDARKEST = 'BLUE_DARKEST'
    TEALLIGHTEST = 'TEAL_LIGHTEST'
    TEALLIGHTER = 'TEAL_LIGHTER'
    TEAL = 'TEAL'
    TEALDARKER = 'TEAL_DARKER'
    TEALDARKEST = 'TEAL_DARKEST'
    GREENLIGHTEST = 'GREEN_LIGHTEST'
    GREENLIGHTER = 'GREEN_LIGHTER'
    GREEN = 'GREEN'
    GREENDARKER = 'GREEN_DARKER'
    GREENDARKEST = 'GREEN_DARKEST'
    LIMELIGHTEST = 'LIME_LIGHTEST'
    LIMELIGHTER = 'LIME_LIGHTER'
    LIME = 'LIME'
    LIMEDARKER = 'LIME_DARKER'
    LIMEDARKEST = 'LIME_DARKEST'
    YELLOWLIGHTEST = 'YELLOW_LIGHTEST'
    YELLOWLIGHTER = 'YELLOW_LIGHTER'
    YELLOW = 'YELLOW'
    YELLOWDARKER = 'YELLOW_DARKER'
    YELLOWDARKEST = 'YELLOW_DARKEST'
    ORANGELIGHTEST = 'ORANGE_LIGHTEST'
    ORANGELIGHTER = 'ORANGE_LIGHTER'
    ORANGE = 'ORANGE'
    ORANGEDARKER = 'ORANGE_DARKER'
    ORANGEDARKEST = 'ORANGE_DARKEST'
    REDLIGHTEST = 'RED_LIGHTEST'
    REDLIGHTER = 'RED_LIGHTER'
    RED = 'RED'
    REDDARKER = 'RED_DARKER'
    REDDARKEST = 'RED_DARKEST'
    MAGENTALIGHTEST = 'MAGENTA_LIGHTEST'
    MAGENTALIGHTER = 'MAGENTA_LIGHTER'
    MAGENTA = 'MAGENTA'
    MAGENTADARKER = 'MAGENTA_DARKER'
    MAGENTADARKEST = 'MAGENTA_DARKEST'


class JiraGroupInput(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    groupName: str


class JiraSingleGroupPickerField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    group: JiraGroupInput


class JiraColorInput(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    name: str


class JiraComponentField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    componentId: int


class JiraSingleSelectField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    option: JiraSelectedOptionField


class JiraLabelsInput(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    name: str


class JiraLabelPropertiesInputJackson1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    color: JiraLabelPropertiesInputJackson1Color = Field(default=None)
    name: str = Field(default=None)


class JiraLabelsField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    bulkEditMultiSelectFieldOption: IssueBulkEditFieldEnum
    fieldId: str
    labelProperties: list[JiraLabelPropertiesInputJackson1] = Field(default=None)
    labels: list[JiraLabelsInput]


class JiraDurationField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    originalEstimateField: str


class JiraColorField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    color: JiraColorInput
    fieldId: str


class JiraUrlField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    url: str


class JiraDateField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    date: JiraDateInput = Field(default=None)
    fieldId: str


class JiraMultipleVersionPickerField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    bulkEditMultiSelectFieldOption: IssueBulkEditFieldEnum
    fieldId: str
    versions: list[JiraVersionField]


class JiraCascadingSelectField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    childOptionValue: JiraSelectedOptionField = Field(default=None)
    fieldId: str
    parentOptionValue: JiraSelectedOptionField


class JiraUserField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str


class JiraMultipleSelectUserPickerField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    users: list[JiraUserField] = Field(default=None)


class JiraRichTextInput(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    adfValue: dict[str, dict[str, Any]] = Field(default=None)


class JiraRichTextField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    richText: JiraRichTextInput


class JiraDateTimeInput(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    formattedDateTime: str


class JiraTimeTrackingField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    timeRemaining: str


class IssueBulkEditPayloadUnnamedModel4(JiraTimeTrackingField):
    pass


class JiraDateTimeField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    dateTime: JiraDateTimeInput
    fieldId: str


class JiraMultiSelectComponentField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    bulkEditMultiSelectFieldOption: IssueBulkEditFieldEnum
    components: list[JiraComponentField]
    fieldId: str


class IssueBulkEditPayloadUnnamedModel1(JiraMultiSelectComponentField):
    pass


class JiraSingleLineTextField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    text: str


class JiraStatusInput(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statusId: str


class JiraSingleSelectUserPickerField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    user: JiraUserField = Field(default=None)


class JiraMultipleSelectField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    options: list[JiraSelectedOptionField]


class JiraMultipleGroupPickerField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    groups: list[JiraGroupInput]


class IssueBulkEditPayloadUnnamedModel3(JiraPriorityField):
    pass


class IssueBulkEditPayloadUnnamedModel2(JiraDurationField):
    pass


class JiraIssueFields(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    cascadingSelectFields: list[JiraCascadingSelectField] = Field(default=None)
    clearableNumberFields: list[JiraNumberField] = Field(default=None)
    colorFields: list[JiraColorField] = Field(default=None)
    datePickerFields: list[JiraDateField] = Field(default=None)
    dateTimePickerFields: list[JiraDateTimeField] = Field(default=None)
    issueType: IssueBulkEditPayloadUnnamedModel = Field(default=None)
    labelsFields: list[JiraLabelsField] = Field(default=None)
    multipleGroupPickerFields: list[JiraMultipleGroupPickerField] = Field(default=None)
    multipleSelectClearableUserPickerFields: list[JiraMultipleSelectUserPickerField] = (
        Field(default=None)
    )
    multipleSelectFields: list[JiraMultipleSelectField] = Field(default=None)
    multipleVersionPickerFields: list[JiraMultipleVersionPickerField] = Field(
        default=None
    )
    multiselectComponents: IssueBulkEditPayloadUnnamedModel1 = Field(default=None)
    originalEstimateField: IssueBulkEditPayloadUnnamedModel2 = Field(default=None)
    priority: IssueBulkEditPayloadUnnamedModel3 = Field(default=None)
    richTextFields: list[JiraRichTextField] = Field(default=None)
    singleGroupPickerFields: list[JiraSingleGroupPickerField] = Field(default=None)
    singleLineTextFields: list[JiraSingleLineTextField] = Field(default=None)
    singleSelectClearableUserPickerFields: list[JiraSingleSelectUserPickerField] = (
        Field(default=None)
    )
    singleSelectFields: list[JiraSingleSelectField] = Field(default=None)
    singleVersionPickerFields: list[JiraSingleVersionPickerField] = Field(default=None)
    status: JiraStatusInput = Field(default=None)
    timeTrackingField: IssueBulkEditPayloadUnnamedModel4 = Field(default=None)
    urlFields: list[JiraUrlField] = Field(default=None)


class IssueBulkEditPayloadUnnamedModel5(JiraIssueFields):
    pass


class IssueBulkEditPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    editedFieldsInput: IssueBulkEditPayloadUnnamedModel5
    selectedActions: list[str]
    selectedIssueIdsOrKeys: list[str]
    sendBulkNotification: bool | None = Field(default=True)


class targetToSourcesMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    inferClassificationDefaults: bool
    inferFieldDefaults: bool
    inferStatusDefaults: bool
    inferSubtaskTypeDefault: bool
    issueIdsOrKeys: list[str] = Field(default=None)
    targetClassification: Any | None = Field(default=None)
    targetMandatoryFields: Any | None = Field(default=None)
    targetStatus: Any | None = Field(default=None)


class IssueBulkMovePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    sendBulkNotification: bool | None = Field(default=True)
    targetToSourcesMapping_: dict[str, targetToSourcesMapping] = Field(
        default=None, alias='targetToSourcesMapping'
    )


class IssueTransitionStatus(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statusId: int = Field(default=None)
    statusName: str = Field(default=None)


class SimplifiedIssueTransitionUnnamedModel(IssueTransitionStatus):
    pass


class SimplifiedIssueTransition(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    to: SimplifiedIssueTransitionUnnamedModel = Field(default=None)
    transitionId: int = Field(default=None)
    transitionName: str = Field(default=None)


class IssueBulkTransitionForWorkflow(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isTransitionsFiltered: bool = Field(default=None)
    issues: list[str] = Field(default=None)
    transitions: list[SimplifiedIssueTransition] = Field(default=None)


class BulkTransitionGetAvailableTransitions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    availableTransitions: list[IssueBulkTransitionForWorkflow] = Field(default=None)
    endingBefore: str = Field(default=None)
    startingAfter: str = Field(default=None)


class BulkTransitionSubmitInput(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    selectedIssueIdsOrKeys: list[str]
    transitionId: str


class IssueBulkTransitionPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    bulkTransitionInputs: list[BulkTransitionSubmitInput]
    sendBulkNotification: bool | None = Field(default=True)


class IssueBulkWatchOrUnwatchPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    selectedIssueIdsOrKeys: list[str]


class BulkOperationProgressStatus(_HtmlReprMixin, str, Enum):
    ENQUEUED = 'ENQUEUED'
    RUNNING = 'RUNNING'
    COMPLETE = 'COMPLETE'
    FAILED = 'FAILED'
    CANCELREQUESTED = 'CANCEL_REQUESTED'
    CANCELLED = 'CANCELLED'
    DEAD = 'DEAD'


class BulkOperationProgress(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    created: datetime = Field(default=None)
    failedAccessibleIssues: dict[str, list[str]] = Field(default=None)
    invalidOrInaccessibleIssueCount: int = Field(default=None)
    processedAccessibleIssues: list[int] = Field(default=None)
    progressPercent: int = Field(default=None)
    started: datetime = Field(default=None)
    status: BulkOperationProgressStatus = Field(default=None)
    submittedBy: User = Field(default=None)
    taskId: str = Field(default=None)
    totalIssueCount: int = Field(default=None)
    updated: datetime = Field(default=None)


class BulkChangelogRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldIds: list[str] = Field(default=None, max_length=10)
    issueIdsOrKeys: list[str] = Field(min_length=1, max_length=1000)
    maxResults: int = Field(default=1000, ge=1.0, le=10000.0)
    nextPageToken: str = Field(default=None)


class ChangelogUnnamedModel(AvatarUrlsBean):
    pass


class UserDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: ChangelogUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class HistoryMetadataParticipant(_HtmlReprMixin, BaseModel):
    avatarUrl: str = Field(default=None)
    displayName: str = Field(default=None)
    displayNameKey: str = Field(default=None)
    id: str = Field(default=None)
    type_: str = Field(default=None, alias='type')
    url: str = Field(default=None)


class ChangelogUnnamedModel2(HistoryMetadataParticipant):
    pass


class ChangelogUnnamedModel4(HistoryMetadataParticipant):
    pass


class ChangelogUnnamedModel3(HistoryMetadataParticipant):
    pass


class HistoryMetadata(_HtmlReprMixin, BaseModel):
    activityDescription: str = Field(default=None)
    activityDescriptionKey: str = Field(default=None)
    actor: ChangelogUnnamedModel2 = Field(default=None)
    cause: ChangelogUnnamedModel3 = Field(default=None)
    description: str = Field(default=None)
    descriptionKey: str = Field(default=None)
    emailDescription: str = Field(default=None)
    emailDescriptionKey: str = Field(default=None)
    extraData: dict[str, str] = Field(default=None)
    generator: ChangelogUnnamedModel4 = Field(default=None)
    type_: str = Field(default=None, alias='type')


class ChangeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    field: str = Field(default=None)
    fieldId: str = Field(default=None)
    fieldtype: str = Field(default=None)
    from_: str = Field(default=None, alias='from')
    fromString: str = Field(default=None)
    to: str = Field(default=None)
    toString: str = Field(default=None)


class ChangelogUnnamedModel1(UserDetails):
    pass


class ChangelogUnnamedModel5(HistoryMetadata):
    pass


class Changelog(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    author: ChangelogUnnamedModel1 = Field(default=None)
    created: datetime = Field(default=None)
    historyMetadata: ChangelogUnnamedModel5 = Field(default=None)
    id: str = Field(default=None)
    items: list[ChangeDetails] = Field(default=None)


class IssueChangeLog(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    changeHistories: list[Changelog] = Field(default=None)
    issueId: str = Field(default=None)


class BulkChangelogResponseBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueChangeLogs: list[IssueChangeLog] = Field(default=None)
    nextPageToken: str = Field(default=None)


class AutoEnum1(_HtmlReprMixin, str, Enum):
    PUBLISHED = 'PUBLISHED'
    ARCHIVED = 'ARCHIVED'
    DRAFT = 'DRAFT'


class AutoEnum2(_HtmlReprMixin, str, Enum):
    RANK = 'rank'
    RANK_1 = '-rank'
    RANK_2 = '+rank'


class DataClassificationTagBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    color: str = Field(default=None)
    description: str = Field(default=None)
    guideline: str = Field(default=None)
    guidelineADF: str = Field(default=None)
    id: str
    name: str = Field(default=None)
    rank: int = Field(default=None)
    status: str


class DataClassificationLevelsBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    classifications: list[DataClassificationTagBean] = Field(default=None)


class IssueCommentListRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    ids: list[int]


class EntityProperty(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    key: str = Field(default=None)
    value: dict[str, Any] = Field(default=None)


class CommentUnnamedModel2(AvatarUrlsBean):
    pass


class CommentUnnamedModel(AvatarUrlsBean):
    pass


class CommentType(_HtmlReprMixin, str, Enum):
    GROUP = 'group'
    ROLE = 'role'


class CommentUserDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: CommentUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class CommentUnnamedModel1(CommentUserDetails):
    pass


class Visibility(_HtmlReprMixin, BaseModel):
    identifier: str | None = Field(default=None)
    type_: CommentType = Field(default=None, alias='type')
    value: str = Field(default=None)


class CommentUserDetails1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: CommentUnnamedModel2 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class CommentUnnamedModel3(CommentUserDetails1):
    pass


class CommentUnnamedModel4(Visibility):
    pass


class Comment(_HtmlReprMixin, BaseModel):
    author: CommentUnnamedModel1 = Field(default=None)
    body: dict[str, Any] = Field(default=None)
    created: datetime = Field(default=None)
    id: str = Field(default=None)
    jsdAuthorCanSeeRequest: bool = Field(default=None)
    jsdPublic: bool = Field(default=None)
    properties: list[EntityProperty] = Field(default=None)
    renderedBody: str = Field(default=None)
    self: str = Field(default=None)
    updateAuthor: CommentUnnamedModel3 = Field(default=None)
    updated: datetime = Field(default=None)
    visibility: CommentUnnamedModel4 = Field(default=None)


class PageBeanComment(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Comment] = Field(default=None)


class PropertyKey(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    key: str = Field(default=None)
    self: str = Field(default=None)


class PropertyKeys(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    keys: list[PropertyKey] = Field(default=None)


class AutoEnum3(_HtmlReprMixin, str, Enum):
    DESCRIPTION = 'description'
    DESCRIPTION_1 = '-description'
    DESCRIPTION_2 = '+description'
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'


class ComponentJsonBean(_HtmlReprMixin, BaseModel):
    ari: str = Field(default=None)
    description: str = Field(default=None)
    id: str = Field(default=None)
    metadata: dict[str, str] = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class PageBean2ComponentJsonBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ComponentJsonBean] = Field(default=None)


class ProjectComponentAssigneeType(_HtmlReprMixin, str, Enum):
    PROJECTDEFAULT = 'PROJECT_DEFAULT'
    COMPONENTLEAD = 'COMPONENT_LEAD'
    PROJECTLEAD = 'PROJECT_LEAD'
    UNASSIGNED = 'UNASSIGNED'


class ProjectComponentUnnamedModel6(AvatarUrlsBean):
    pass


class ProjectComponentUnnamedModel(AvatarUrlsBean):
    pass


class ProjectComponentUnnamedModel1(SimpleListWrapperGroupName):
    pass


class ProjectComponentUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: ProjectComponentUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: ProjectComponentUnnamedModel1 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class ProjectComponentUnnamedModel4(SimpleListWrapperGroupName):
    pass


class ProjectComponentUnnamedModel3(AvatarUrlsBean):
    pass


class ProjectComponentUser1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: ProjectComponentUnnamedModel3 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: ProjectComponentUnnamedModel4 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class ProjectComponentUnnamedModel5(ProjectComponentUser1):
    pass


class ProjectComponentUnnamedModel2(ProjectComponentUser):
    pass


class ProjectComponentUnnamedModel7(SimpleListWrapperGroupName):
    pass


class ProjectComponentUser2(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: ProjectComponentUnnamedModel6 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: ProjectComponentUnnamedModel7 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class ProjectComponentUnnamedModel8(ProjectComponentUser2):
    pass


class ProjectComponent(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    ari: str = Field(default=None)
    assignee: ProjectComponentUnnamedModel2 = Field(default=None)
    assigneeType: ProjectComponentAssigneeType = Field(default=None)
    description: str = Field(default=None)
    id: str = Field(default=None)
    isAssigneeTypeValid: bool = Field(default=None)
    lead: ProjectComponentUnnamedModel5 = Field(default=None)
    leadAccountId: str = Field(default=None, max_length=128)
    leadUserName: str = Field(default=None)
    metadata: dict[str, str] = Field(default=None)
    name: str = Field(default=None)
    project: str = Field(default=None)
    projectId: int = Field(default=None)
    realAssignee: ProjectComponentUnnamedModel8 = Field(default=None)
    realAssigneeType: ProjectComponentAssigneeType = Field(default=None)
    self: str = Field(default=None)


class ComponentIssuesCount(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueCount: int = Field(default=None)
    self: str = Field(default=None)


class FieldAssociationSchemeMatchedFilters(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectIds: list[int] = Field(default=None)
    query: str = Field(default=None)


class FieldAssociationSchemeLinksBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    associations: str = Field(default=None)
    projects: str = Field(default=None)


class GetFieldAssociationSchemeResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    fieldsCount: int = Field(default=None)
    id: int = Field(default=None)
    isDefault: bool = Field(default=None)
    links: FieldAssociationSchemeLinksBean = Field(default=None)
    matchedFilters: FieldAssociationSchemeMatchedFilters = Field(default=None)
    name: str = Field(default=None)


class PageBean2GetFieldAssociationSchemeResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[GetFieldAssociationSchemeResponse] = Field(default=None)


class CreateFieldAssociationSchemeRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str


class CreateFieldAssociationSchemeLinksBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    associations: str = Field(default=None)
    projects: str = Field(default=None)


class CreateFieldAssociationSchemeResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: int = Field(default=None)
    links: CreateFieldAssociationSchemeLinksBean = Field(default=None)
    name: str = Field(default=None)


class UpdateFieldAssociationsRequestItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    restrictedToWorkTypes: list[int] = Field(default=None)
    schemeIds: list[int]


class FieldSchemeToFieldsPartialFailure(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    error: str = Field(default=None)
    fieldId: str
    schemeId: int
    success: bool
    workTypeIds: list[int]


class FieldSchemeToFieldsResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    results: list[FieldSchemeToFieldsPartialFailure]


class RemoveFieldAssociationsRequestItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    schemeIds: list[int]


class MinimalFieldSchemeToFieldsPartialFailure(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    error: str = Field(default=None)
    fieldId: str
    schemeId: int
    success: bool


class MinimalFieldSchemeToFieldsResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    results: list[MinimalFieldSchemeToFieldsPartialFailure]


class FieldsSchemeItemParameterRendererType(_HtmlReprMixin, str, Enum):
    JIRATEXTRENDERER = 'jira-text-renderer'
    ATLASSIANWIKIRENDERER = 'atlassian-wiki-renderer'


class FieldsSchemeItemParameter(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    isRequired: bool = Field(default=None)
    rendererType: FieldsSchemeItemParameterRendererType = Field(default=None)


class FieldsSchemeItemWorkTypeParameter(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    isRequired: bool = Field(default=None)
    rendererType: FieldsSchemeItemParameterRendererType = Field(default=None)
    workTypeId: int = Field(default=None)


class UpdateFieldSchemeParametersRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    parameters: FieldsSchemeItemParameter = Field(default=None)
    schemeIds: list[int] = Field(default=None)
    workTypeParameters: list[FieldsSchemeItemWorkTypeParameter] = Field(default=None)


class UpdateFieldSchemeParametersPartialFailure(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    error: str = Field(default=None)
    fieldId: str
    schemeId: int
    success: bool
    workTypeId: int = Field(default=None)


class UpdateFieldSchemeParametersResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    results: list[UpdateFieldSchemeParametersPartialFailure]


class ParameterRemovalDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    parameters: list[str] = Field(default=None)
    schemeId: int = Field(default=None)
    workTypeIds: list[int] = Field(default=None)


class RemoveFieldParametersResultError(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    code: str = Field(default=None)
    message: str = Field(default=None)


class SuccessOrErrorResults(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    error: RemoveFieldParametersResultError = Field(default=None)
    fieldId: str = Field(default=None)
    schemeId: int = Field(default=None)
    success: bool = Field(default=None)
    workTypeIds: list[int] = Field(default=None)


class RemoveFieldParametersResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    results: list[SuccessOrErrorResults] = Field(default=None)


class GetProjectsWithFieldSchemesResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectId: int = Field(default=None)
    schemeId: int = Field(default=None)


class PageBean2GetProjectsWithFieldSchemesResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[GetProjectsWithFieldSchemesResponse] = Field(default=None)


class FieldSchemeToProjectsRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectIds: list[int]


class FieldSchemeToProjectsPartialFailure(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    error: str = Field(default=None)
    projectId: int
    schemeId: int
    success: bool


class FieldSchemeToProjectsResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    results: list[FieldSchemeToProjectsPartialFailure]


class FieldAssociationSchemeLinks(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    associations: str = Field(default=None)
    projects: str = Field(default=None)


class GetFieldAssociationSchemeByIdResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    fieldsCount: int = Field(default=None)
    id: str = Field(default=None)
    isDefault: bool = Field(default=None)
    links: FieldAssociationSchemeLinks = Field(default=None)
    name: str = Field(default=None)


class UpdateFieldAssociationSchemeRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)


class UpdateFieldAssociationSchemeLinksBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    associations: str = Field(default=None)
    projects: str = Field(default=None)


class UpdateFieldAssociationSchemeResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: int = Field(default=None)
    links: UpdateFieldAssociationSchemeLinksBean = Field(default=None)
    name: str = Field(default=None)


class DeleteFieldAssociationSchemeResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    deleted: bool = Field(default=None)
    id: str = Field(default=None)


class SearchResultWorkTypeParameters(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    isRequired: bool = Field(default=None)
    rendererType: str = Field(default=None)
    workTypeId: str = Field(default=None)


class SearchResultFieldParameters(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    isRequired: bool = Field(default=None)
    rendererType: str = Field(default=None)


class FieldAssociationSchemeFieldSearchResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    allowedOperations: list[str] = Field(default=None)
    fieldId: str = Field(default=None)
    parameters: SearchResultFieldParameters = Field(default=None)
    restrictedToWorkTypes: list[str] = Field(default=None)
    workTypeParameters: list[SearchResultWorkTypeParameters] = Field(default=None)


class PageBean2FieldAssociationSchemeFieldSearchResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FieldAssociationSchemeFieldSearchResult] = Field(default=None)


class WorkTypeParameters(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    isRequired: bool
    rendererType: str = Field(default=None)
    workTypeId: int


class FieldAssociationParameters(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    isRequired: bool
    rendererType: str = Field(default=None)


class GetFieldAssociationParametersResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str
    parameters: FieldAssociationParameters = Field(default=None)
    workTypeParameters: list[WorkTypeParameters] = Field(default=None)


class FieldAssociationSchemeProjectSearchResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: dict[str, str] = Field(default=None)
    deleted: bool = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)


class PageBean2FieldAssociationSchemeProjectSearchResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FieldAssociationSchemeProjectSearchResult] = Field(default=None)


class ConfigurationDefaultUnit(_HtmlReprMixin, str, Enum):
    MINUTE = 'minute'
    HOUR = 'hour'
    DAY = 'day'
    WEEK = 'week'


class ConfigurationTimeFormat(_HtmlReprMixin, str, Enum):
    PRETTY = 'pretty'
    DAYS = 'days'
    HOURS = 'hours'


class TimeTrackingConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultUnit: ConfigurationDefaultUnit
    timeFormat: ConfigurationTimeFormat
    workingDaysPerWeek: float
    workingHoursPerDay: float


class ConfigurationUnnamedModel(TimeTrackingConfiguration):
    pass


class Configuration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    attachmentsEnabled: bool = Field(default=None)
    issueLinkingEnabled: bool = Field(default=None)
    subTasksEnabled: bool = Field(default=None)
    timeTrackingConfiguration: ConfigurationUnnamedModel = Field(default=None)
    timeTrackingEnabled: bool = Field(default=None)
    unassignedIssuesAllowed: bool = Field(default=None)
    votingEnabled: bool = Field(default=None)
    watchingEnabled: bool = Field(default=None)


class TimeTrackingProvider(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    key: str
    name: str = Field(default=None)
    url: str = Field(default=None)


class CustomFieldOption(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    self: str = Field(default=None)
    value: str = Field(default=None)


class AutoEnum4(_HtmlReprMixin, str, Enum):
    MY = 'my'
    FAVOURITE = 'favourite'


class UpdatedProjectCategory(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class SharePermissionUnnamedModel17(AvatarUrlsBean):
    pass


class SharePermissionUnnamedModel18(UpdatedProjectCategory):
    pass


class IssueTypeDetailsProjectTypeKey(_HtmlReprMixin, str, Enum):
    SOFTWARE = 'software'
    SERVICEDESK = 'service_desk'
    BUSINESS = 'business'


class SharePermissionProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: SharePermissionUnnamedModel17 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: SharePermissionUnnamedModel18 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class SharePermissionUnnamedModel19(SharePermissionProjectDetails):
    pass


class IssueTypeDetailsType(_HtmlReprMixin, str, Enum):
    PROJECT = 'PROJECT'
    TEMPLATE = 'TEMPLATE'


class SharePermissionScope(_HtmlReprMixin, BaseModel):
    project: SharePermissionUnnamedModel19 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class SharePermissionUnnamedModel20(SharePermissionScope):
    pass


class SharePermissionUnnamedModel5(AvatarUrlsBean):
    pass


class IssueTypeDetailsUnnamedModel(AvatarUrlsBean):
    pass


class SharePermissionUnnamedModel(GroupName):
    pass


class SimplifiedHierarchyLevel(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    aboveLevelId: int = Field(default=None)
    belowLevelId: int = Field(default=None)
    externalUuid: UUID = Field(default=None)
    hierarchyLevelNumber: int = Field(default=None)
    id: int = Field(default=None)
    issueTypeIds: list[int] = Field(default=None)
    level: int = Field(default=None)
    name: str = Field(default=None)
    projectConfigurationId: int = Field(default=None)


class Hierarchy(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    baseLevelId: int = Field(default=None)
    levels: list[SimplifiedHierarchyLevel] = Field(default=None)


class ProjectCategory(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class SharePermissionUnnamedModel15(ProjectCategory):
    pass


class UserBeanAvatarUrls(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    field_16x16: str = Field(default=None, alias='16x16')
    field_24x24: str = Field(default=None, alias='24x24')
    field_32x32: str = Field(default=None, alias='32x32')
    field_48x48: str = Field(default=None, alias='48x48')


class SharePermissionUnnamedModel22(UserBeanAvatarUrls):
    pass


class UserBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    active: bool = Field(default=None)
    avatarUrls: SharePermissionUnnamedModel22 = Field(default=None)
    displayName: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class SharePermissionAssigneeType(_HtmlReprMixin, str, Enum):
    PROJECTLEAD = 'PROJECT_LEAD'
    UNASSIGNED = 'UNASSIGNED'


class DashboardUnnamedModel(UserBeanAvatarUrls):
    pass


class DashboardUserBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    active: bool = Field(default=None)
    avatarUrls: DashboardUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class ProjectPermissions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    canEdit: bool = Field(default=None)


class SharePermissionUnnamedModel11(AvatarUrlsBean):
    pass


class ProjectInsight(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    lastIssueUpdateTime: datetime = Field(default=None)
    totalIssueCount: int = Field(default=None)


class SharePermissionUnnamedModel8(ProjectInsight):
    pass


class IssueTypeDetailsUnnamedModel1(UpdatedProjectCategory):
    pass


class ProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: IssueTypeDetailsUnnamedModel = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: IssueTypeDetailsUnnamedModel1 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class SharePermissionUnnamedModel6(SimpleListWrapperGroupName):
    pass


class IssueTypeDetailsUnnamedModel2(ProjectDetails):
    pass


class Scope(_HtmlReprMixin, BaseModel):
    project: IssueTypeDetailsUnnamedModel2 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class VersionIssuesStatus(_HtmlReprMixin, BaseModel):
    done: int = Field(default=None)
    inProgress: int = Field(default=None)
    toDo: int = Field(default=None)
    unmapped: int = Field(default=None)


class SharePermissionUnnamedModel1(AvatarUrlsBean):
    pass


class SharePermissionUnnamedModel2(SimpleListWrapperGroupName):
    pass


class SharePermissionUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: SharePermissionUnnamedModel1 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: SharePermissionUnnamedModel2 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class SharePermissionUnnamedModel3(SharePermissionUser):
    pass


class ProjectLandingPageInfo(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    attributes: dict[str, str] = Field(default=None)
    boardId: int = Field(default=None)
    boardName: str = Field(default=None)
    projectKey: str = Field(default=None)
    projectType: str = Field(default=None)
    queueCategory: str = Field(default=None)
    queueId: int = Field(default=None)
    queueName: str = Field(default=None)
    simpleBoard: bool = Field(default=None)
    simplified: bool = Field(default=None)
    url: str = Field(default=None)


class SharePermissionUnnamedModel9(Hierarchy):
    pass


class IssueTypeDetailsUnnamedModel3(Scope):
    pass


class IssueTypeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    entityId: UUID = Field(default=None)
    hierarchyLevel: int = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: IssueTypeDetailsUnnamedModel3 = Field(default=None)
    self: str = Field(default=None)
    subtask: bool = Field(default=None)


class SharePermissionUser1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: SharePermissionUnnamedModel5 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: SharePermissionUnnamedModel6 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class SharePermissionUnnamedModel7(SharePermissionUser1):
    pass


class VersionUnnamedModel(VersionIssuesStatus):
    pass


class SharePermissionUnnamedModel12(SimpleListWrapperGroupName):
    pass


class SimpleLink(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    href: str = Field(default=None)
    iconClass: str = Field(default=None)
    id: str = Field(default=None)
    label: str = Field(default=None)
    styleClass: str = Field(default=None)
    title: str = Field(default=None)
    weight: int = Field(default=None)


class VersionApprover(_HtmlReprMixin, BaseModel):
    accountId: str = Field(default=None)
    declineReason: str = Field(default=None)
    description: str = Field(default=None)
    status: str = Field(default=None)


class Version(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    approvers: list[VersionApprover] = Field(default=None)
    archived: bool = Field(default=None)
    description: str = Field(default=None)
    driver: str = Field(default=None)
    expand: str = Field(default=None)
    id: str = Field(default=None)
    issuesStatusForFixVersion: VersionUnnamedModel = Field(default=None)
    moveUnfixedIssuesTo: str = Field(default=None)
    name: str = Field(default=None)
    operations: list[SimpleLink] = Field(default=None)
    overdue: bool = Field(default=None)
    project: str = Field(default=None)
    projectId: int = Field(default=None)
    releaseDate: date = Field(default=None)
    released: bool = Field(default=None)
    self: str = Field(default=None)
    startDate: date = Field(default=None)
    userReleaseDate: str = Field(default=None)
    userStartDate: str = Field(default=None)


class SharePermissionStyle(_HtmlReprMixin, str, Enum):
    CLASSIC = 'classic'
    NEXTGEN = 'next-gen'


class SharePermissionUser2(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: SharePermissionUnnamedModel11 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: SharePermissionUnnamedModel12 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class SharePermissionUnnamedModel4(AvatarUrlsBean):
    pass


class SharePermissionUnnamedModel13(SharePermissionUser2):
    pass


class SharePermissionUnnamedModel14(ProjectPermissions):
    pass


class SharePermissionUnnamedModel10(ProjectLandingPageInfo):
    pass


class Project(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    archived: bool = Field(default=None)
    archivedBy: SharePermissionUnnamedModel3 = Field(default=None)
    archivedDate: datetime = Field(default=None)
    assigneeType: SharePermissionAssigneeType = Field(default=None)
    avatarUrls: SharePermissionUnnamedModel4 = Field(default=None)
    components: list[ProjectComponent] = Field(default=None)
    deleted: bool = Field(default=None)
    deletedBy: SharePermissionUnnamedModel7 = Field(default=None)
    deletedDate: datetime = Field(default=None)
    description: str = Field(default=None)
    email: str = Field(default=None)
    expand: str = Field(default=None)
    favourite: bool = Field(default=None)
    id: str = Field(default=None)
    insight: SharePermissionUnnamedModel8 = Field(default=None)
    isPrivate: bool = Field(default=None)
    issueTypeHierarchy: SharePermissionUnnamedModel9 = Field(default=None)
    issueTypes: list[IssueTypeDetails] = Field(default=None)
    key: str = Field(default=None)
    landingPageInfo: SharePermissionUnnamedModel10 = Field(default=None)
    lead: SharePermissionUnnamedModel13 = Field(default=None)
    name: str = Field(default=None)
    permissions: SharePermissionUnnamedModel14 = Field(default=None)
    projectCategory: SharePermissionUnnamedModel15 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    properties: dict[str, dict[str, Any]] = Field(default=None)
    retentionTillDate: datetime = Field(default=None)
    roles: dict[str, str] = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)
    style: SharePermissionStyle = Field(default=None)
    url: str = Field(default=None)
    uuid: UUID = Field(default=None)
    versions: list[Version] = Field(default=None)


class ProjectRoleUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)


class SharePermissionType(_HtmlReprMixin, str, Enum):
    USER = 'user'
    GROUP = 'group'
    PROJECT = 'project'
    PROJECTROLE = 'projectRole'
    GLOBAL = 'global'
    LOGGEDIN = 'loggedin'
    AUTHENTICATED = 'authenticated'
    PROJECTUNKNOWN = 'project-unknown'


class RoleActorUnnamedModel1(ProjectRoleUser):
    pass


class ProjectRoleGroup(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    displayName: str = Field(default=None)
    groupId: str = Field(default=None)
    name: str = Field(default=None)


class RoleActorUnnamedModel(ProjectRoleGroup):
    pass


class RoleActorType(_HtmlReprMixin, str, Enum):
    ATLASSIANGROUPROLEACTOR = 'atlassian-group-role-actor'
    ATLASSIANUSERROLEACTOR = 'atlassian-user-role-actor'


class RoleActor(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    actorGroup: RoleActorUnnamedModel = Field(default=None)
    actorUser: RoleActorUnnamedModel1 = Field(default=None)
    avatarUrl: str = Field(default=None)
    displayName: str = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)
    type_: RoleActorType = Field(default=None, alias='type')


class ProjectRole(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    actors: list[RoleActor] = Field(default=None)
    admin: bool = Field(default=None)
    currentUserRole: bool = Field(default=None)
    default: bool = Field(default=None)
    description: str = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)
    roleConfigurable: bool = Field(default=None)
    scope: SharePermissionUnnamedModel20 = Field(default=None)
    self: str = Field(default=None)
    translatedName: str = Field(default=None)


class SharePermissionUnnamedModel23(UserBean):
    pass


class SharePermissionUnnamedModel21(ProjectRole):
    pass


class SharePermissionUnnamedModel16(Project):
    pass


class SharePermission(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    group: SharePermissionUnnamedModel = Field(default=None)
    id: int = Field(default=None)
    project: SharePermissionUnnamedModel16 = Field(default=None)
    role: SharePermissionUnnamedModel21 = Field(default=None)
    type_: SharePermissionType = Field(alias='type')
    user: SharePermissionUnnamedModel23 = Field(default=None)


class DashboardUnnamedModel1(DashboardUserBean):
    pass


class Dashboard(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    automaticRefreshMs: int = Field(default=None)
    description: str = Field(default=None)
    editPermissions: list[SharePermission] = Field(default=None)
    id: str = Field(default=None)
    isFavourite: bool = Field(default=None)
    isWritable: bool = Field(default=None)
    name: str = Field(default=None)
    owner: DashboardUnnamedModel1 = Field(default=None)
    popularity: int = Field(default=None)
    rank: int = Field(default=None)
    self: str = Field(default=None)
    sharePermissions: list[SharePermission] = Field(default=None)
    systemDashboard: bool = Field(default=None)
    view: str = Field(default=None)


class PageOfDashboards(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    dashboards: list[Dashboard] = Field(default=None)
    maxResults: int = Field(default=None)
    next: str = Field(default=None)
    prev: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)


class DashboardDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    editPermissions: list[SharePermission]
    name: str
    sharePermissions: list[SharePermission]


class BulkEditShareableEntityRequestAction(_HtmlReprMixin, str, Enum):
    CHANGEOWNER = 'changeOwner'
    CHANGEPERMISSION = 'changePermission'
    ADDPERMISSION = 'addPermission'
    REMOVEPERMISSION = 'removePermission'


class PermissionDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    editPermissions: list[SharePermission]
    sharePermissions: list[SharePermission]


class BulkEditShareableEntityRequestUnnamedModel1(PermissionDetails):
    pass


class BulkChangeOwnerDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    autofixName: bool
    newOwner: str


class BulkEditShareableEntityRequestUnnamedModel(BulkChangeOwnerDetails):
    pass


class BulkEditShareableEntityRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    action: BulkEditShareableEntityRequestAction
    changeOwnerDetails: BulkEditShareableEntityRequestUnnamedModel = Field(default=None)
    entityIds: list[int]
    extendAdminPermissions: bool = Field(default=None)
    permissionDetails: BulkEditShareableEntityRequestUnnamedModel1 = Field(default=None)


class BulkEditActionError(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errorMessages: list[str]
    errors: dict[str, str]


class BulkEditShareableEntityResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    action: BulkEditShareableEntityRequestAction
    entityErrors: dict[str, BulkEditActionError] = Field(default=None)


class AvailableDashboardGadget(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    moduleKey: str = Field(default=None)
    title: str
    uri: str = Field(default=None)


class AvailableDashboardGadgetsResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    gadgets: list[AvailableDashboardGadget]


class AutoEnum5(_HtmlReprMixin, str, Enum):
    DESCRIPTION = 'description'
    DESCRIPTION_1 = '-description'
    DESCRIPTION_2 = '+description'
    FAVORITECOUNT = 'favorite_count'
    FAVORITECOUNT_1 = '-favorite_count'
    FAVORITECOUNT_2 = '+favorite_count'
    ID = 'id'
    ID_1 = '-id'
    ID_2 = '+id'
    ISFAVORITE = 'is_favorite'
    ISFAVORITE_1 = '-is_favorite'
    ISFAVORITE_2 = '+is_favorite'
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'
    OWNER = 'owner'
    OWNER_1 = '-owner'
    OWNER_2 = '+owner'


class AutoEnum6(_HtmlReprMixin, str, Enum):
    ACTIVE = 'active'
    ARCHIVED = 'archived'
    DELETED = 'deleted'


class PageBeanDashboard(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Dashboard] = Field(default=None)


class DashboardGadgetPosition(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    The_column_position_of_the_gadget: int = Field(
        alias='The column position of the gadget.'
    )
    The_row_position_of_the_gadget: int = Field(alias='The row position of the gadget.')


class DashboardGadgetUnnamedModel(DashboardGadgetPosition):
    pass


class DashboardGadgetColor(_HtmlReprMixin, str, Enum):
    BLUE = 'blue'
    RED = 'red'
    YELLOW = 'yellow'
    GREEN = 'green'
    CYAN = 'cyan'
    PURPLE = 'purple'
    GRAY = 'gray'
    WHITE = 'white'


class DashboardGadget(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    color: DashboardGadgetColor
    id: int
    moduleKey: str = Field(default=None)
    position: DashboardGadgetUnnamedModel
    title: str
    uri: str = Field(default=None)


class DashboardGadgetResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    gadgets: list[DashboardGadget]


class DashboardGadgetSettingsUnnamedModel(DashboardGadgetPosition):
    pass


class DashboardGadgetSettings(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    color: str = Field(default=None)
    ignoreUriAndModuleKeyValidation: bool = Field(default=None)
    moduleKey: str = Field(default=None)
    position: DashboardGadgetSettingsUnnamedModel = Field(default=None)
    title: str = Field(default=None)
    uri: str = Field(default=None)


class DashboardGadgetUpdateRequestUnnamedModel(DashboardGadgetPosition):
    pass


class DashboardGadgetUpdateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    color: str = Field(default=None)
    position: DashboardGadgetUpdateRequestUnnamedModel = Field(default=None)
    title: str = Field(default=None)


class WorkspaceDataPolicy(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    anyContentBlocked: bool = Field(default=None)


class ProjectDataPolicy(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    anyContentBlocked: bool = Field(default=None)


class ProjectWithDataPolicyUnnamedModel(ProjectDataPolicy):
    pass


class ProjectWithDataPolicy(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    dataPolicy: ProjectWithDataPolicyUnnamedModel = Field(default=None)
    id: int = Field(default=None)


class ProjectDataPolicies(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectDataPolicies: list[ProjectWithDataPolicy] = Field(default=None)


class IssueEvent(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(default=None)
    name: str = Field(default=None)


class AutoEnum7(_HtmlReprMixin, str, Enum):
    SYNTAX = 'syntax'
    TYPE = 'type'
    COMPLEXITY = 'complexity'


class JiraExpressionForAnalysis(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contextVariables: dict[str, str] = Field(default=None)
    expressions: list[str]


class JiraExpressionComplexity(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expensiveOperations: str
    variables: dict[str, str] = Field(default=None)


class JiraExpressionValidationErrorType(_HtmlReprMixin, str, Enum):
    SYNTAX = 'syntax'
    TYPE = 'type'
    OTHER = 'other'


class JiraExpressionValidationError(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    column: int = Field(default=None)
    expression: str = Field(default=None)
    line: int = Field(default=None)
    message: str
    type_: JiraExpressionValidationErrorType = Field(alias='type')


class JiraExpressionAnalysis(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    complexity: JiraExpressionComplexity = Field(default=None)
    errors: list[JiraExpressionValidationError] = Field(default=None)
    expression: str
    type_: str = Field(default=None, alias='type')
    valid: bool


class JiraExpressionsAnalysis(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    results: list[JiraExpressionAnalysis]


class JiraExpressionEvalRequestBeanValidation(_HtmlReprMixin, str, Enum):
    STRICT = 'strict'
    WARN = 'warn'
    NONE = 'none'


class JexpJqlIssues(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    maxResults: int = Field(default=None)
    query: str = Field(default=None)
    startAt: int = Field(default=None)
    validation: JiraExpressionEvalRequestBeanValidation = Field(default='strict')


class JiraExpressionEvalRequestBeanUnnamedModel1(JexpJqlIssues):
    pass


class IdOrKeyBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(default=None)
    key: str = Field(default=None)


class JiraExpressionEvalRequestBeanUnnamedModel3(IdOrKeyBean):
    pass


class JexpIssues(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    jql: JiraExpressionEvalRequestBeanUnnamedModel1 = Field(default=None)


class JiraExpressionEvalRequestBeanUnnamedModel2(JexpIssues):
    pass


class JiraExpressionEvalRequestBeanUnnamedModel(IdOrKeyBean):
    pass


class JiraExpressionEvalContextBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    board: int = Field(default=None)
    custom: list[UserContextVariable | IssueContextVariable | JsonContextVariable] = (
        Field(default=None)
    )
    customerRequest: int = Field(default=None)
    issue: JiraExpressionEvalRequestBeanUnnamedModel = Field(default=None)
    issues: JiraExpressionEvalRequestBeanUnnamedModel2 = Field(default=None)
    project: JiraExpressionEvalRequestBeanUnnamedModel3 = Field(default=None)
    serviceDesk: int = Field(default=None)
    sprint: int = Field(default=None)


class JiraExpressionEvalRequestBeanUnnamedModel4(JiraExpressionEvalContextBean):
    pass


class JiraExpressionEvalRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    context: JiraExpressionEvalRequestBeanUnnamedModel4 = Field(default=None)
    expression: str


class UserContextVariable(_HtmlReprMixin, BaseModel):
    accountId: str
    type_: str = Field(alias='type')


class IssueContextVariable(_HtmlReprMixin, BaseModel):
    id: int = Field(default=None)
    key: str = Field(default=None)
    type_: str = Field(alias='type')


class JsonContextVariable(_HtmlReprMixin, BaseModel):
    type_: str = Field(alias='type')
    value: dict[str, Any] = Field(default=None)


class JiraExpressionsComplexityValueBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    limit: int
    value: int


class JiraExpressionResultUnnamedModel(JiraExpressionsComplexityValueBean):
    pass


class JiraExpressionResultUnnamedModel3(JiraExpressionsComplexityValueBean):
    pass


class JiraExpressionResultUnnamedModel1(JiraExpressionsComplexityValueBean):
    pass


class JiraExpressionResultUnnamedModel2(JiraExpressionsComplexityValueBean):
    pass


class JiraExpressionsComplexityBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    beans: JiraExpressionResultUnnamedModel
    expensiveOperations: JiraExpressionResultUnnamedModel1
    primitiveValues: JiraExpressionResultUnnamedModel2
    steps: JiraExpressionResultUnnamedModel3


class JiraExpressionResultUnnamedModel4(JiraExpressionsComplexityBean):
    pass


class IssuesJqlMetaDataBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    count: int
    maxResults: int
    startAt: int
    totalCount: int
    validationWarnings: list[str] = Field(default=None)


class IssuesMetaBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    jql: IssuesJqlMetaDataBean = Field(default=None)


class JiraExpressionResultUnnamedModel5(IssuesMetaBean):
    pass


class JiraExpressionEvaluationMetaDataBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    complexity: JiraExpressionResultUnnamedModel4 = Field(default=None)
    issues: JiraExpressionResultUnnamedModel5 = Field(default=None)


class JiraExpressionResultUnnamedModel6(JiraExpressionEvaluationMetaDataBean):
    pass


class JiraExpressionResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    meta: JiraExpressionResultUnnamedModel6 = Field(default=None)
    value: dict[str, Any]


class JexpEvaluateCtxJqlIssues(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    maxResults: int = Field(default=None)
    nextPageToken: str = Field(default=None)
    query: str = Field(default=None)


class JiraExpressionEvaluateRequestBeanUnnamedModel(IdOrKeyBean):
    pass


class JiraExpressionEvaluateRequestBeanUnnamedModel1(JexpEvaluateCtxJqlIssues):
    pass


class JexpEvaluateCtxIssues(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    jql: JiraExpressionEvaluateRequestBeanUnnamedModel1 = Field(default=None)


class JiraExpressionEvaluateRequestBeanUnnamedModel2(JexpEvaluateCtxIssues):
    pass


class JiraExpressionEvaluateRequestBeanUnnamedModel3(IdOrKeyBean):
    pass


class JiraExpressionEvaluateContextBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    board: int = Field(default=None)
    custom: list[UserContextVariable | IssueContextVariable | JsonContextVariable] = (
        Field(default=None)
    )
    customerRequest: int = Field(default=None)
    issue: JiraExpressionEvaluateRequestBeanUnnamedModel = Field(default=None)
    issues: JiraExpressionEvaluateRequestBeanUnnamedModel2 = Field(default=None)
    project: JiraExpressionEvaluateRequestBeanUnnamedModel3 = Field(default=None)
    serviceDesk: int = Field(default=None)
    sprint: int = Field(default=None)


class JiraExpressionEvaluateRequestBeanUnnamedModel4(JiraExpressionEvaluateContextBean):
    pass


class JiraExpressionEvaluateRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    context: JiraExpressionEvaluateRequestBeanUnnamedModel4 = Field(default=None)
    expression: str


class JExpEvaluateIssuesJqlMetaDataBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    nextPageToken: str


class JExpEvaluateJiraExpressionResultBeanUnnamedModel2(
    JiraExpressionsComplexityValueBean
):
    pass


class JExpEvaluateIssuesMetaBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    jql: JExpEvaluateIssuesJqlMetaDataBean = Field(default=None)


class JExpEvaluateJiraExpressionResultBeanUnnamedModel5(JExpEvaluateIssuesMetaBean):
    pass


class JExpEvaluateJiraExpressionResultBeanUnnamedModel3(
    JiraExpressionsComplexityValueBean
):
    pass


class JExpEvaluateJiraExpressionResultBeanUnnamedModel1(
    JiraExpressionsComplexityValueBean
):
    pass


class JExpEvaluateJiraExpressionResultBeanUnnamedModel(
    JiraExpressionsComplexityValueBean
):
    pass


class JExpEvaluateJiraExpressionResultBeanJiraExpressionsComplexityBean(
    _HtmlReprMixin, BaseModel
):
    model_config = {'extra': 'forbid'}
    beans: JExpEvaluateJiraExpressionResultBeanUnnamedModel
    expensiveOperations: JExpEvaluateJiraExpressionResultBeanUnnamedModel1
    primitiveValues: JExpEvaluateJiraExpressionResultBeanUnnamedModel2
    steps: JExpEvaluateJiraExpressionResultBeanUnnamedModel3


class JExpEvaluateJiraExpressionResultBeanUnnamedModel4(
    JExpEvaluateJiraExpressionResultBeanJiraExpressionsComplexityBean
):
    pass


class JExpEvaluateMetaDataBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    complexity: JExpEvaluateJiraExpressionResultBeanUnnamedModel4 = Field(default=None)
    issues: JExpEvaluateJiraExpressionResultBeanUnnamedModel5 = Field(default=None)


class JExpEvaluateJiraExpressionResultBeanUnnamedModel6(JExpEvaluateMetaDataBean):
    pass


class JExpEvaluateJiraExpressionResultBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    meta: JExpEvaluateJiraExpressionResultBeanUnnamedModel6 = Field(default=None)
    value: dict[str, Any]


class JsonTypeBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    configuration: dict[str, dict[str, Any]] = Field(default=None)
    custom: str = Field(default=None)
    customId: int = Field(default=None)
    items: str = Field(default=None)
    system: str = Field(default=None)
    type_: str = Field(alias='type')


class FieldDetailsUnnamedModel(JsonTypeBean):
    pass


class FieldDetailsUnnamedModel1(AvatarUrlsBean):
    pass


class FieldDetailsUnnamedModel2(UpdatedProjectCategory):
    pass


class FieldDetailsProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: FieldDetailsUnnamedModel1 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: FieldDetailsUnnamedModel2 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class FieldDetailsUnnamedModel3(FieldDetailsProjectDetails):
    pass


class FieldDetailsScope(_HtmlReprMixin, BaseModel):
    project: FieldDetailsUnnamedModel3 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class FieldDetailsUnnamedModel4(FieldDetailsScope):
    pass


class FieldDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    clauseNames: list[str] = Field(default=None)
    custom: bool = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    navigable: bool = Field(default=None)
    orderable: bool = Field(default=None)
    schema: FieldDetailsUnnamedModel = Field(default=None)
    scope: FieldDetailsUnnamedModel4 = Field(default=None)
    searchable: bool = Field(default=None)


class CustomFieldDefinitionJsonBeanSearcherKey(_HtmlReprMixin, str, Enum):
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESCASCADINGSELECTSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:cascadingselectsearcher'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESDATERANGE = (
        'com.atlassian.jira.plugin.system.customfieldtypes:daterange'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESDATETIMERANGE = (
        'com.atlassian.jira.plugin.system.customfieldtypes:datetimerange'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESEXACTNUMBER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:exactnumber'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESEXACTTEXTSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:exacttextsearcher'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESGROUPPICKERSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:grouppickersearcher'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESLABELSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:labelsearcher'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESMULTISELECTSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:multiselectsearcher'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESNUMBERRANGE = (
        'com.atlassian.jira.plugin.system.customfieldtypes:numberrange'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESPROJECTSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:projectsearcher'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESTEXTSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:textsearcher'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESUSERPICKERGROUPSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:userpickergroupsearcher'
    )
    COMATLASSIANJIRAPLUGINSYSTEMCUSTOMFIELDTYPESVERSIONSEARCHER = (
        'com.atlassian.jira.plugin.system.customfieldtypes:versionsearcher'
    )


class CustomFieldDefinitionJsonBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str
    searcherKey: CustomFieldDefinitionJsonBeanSearcherKey = Field(default=None)
    type_: str = Field(alias='type')


class AssociationContextObject(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    identifier: dict[str, Any] = Field(default=None)
    type_: str = Field(alias='type')


class FieldIdentifierObject(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    identifier: dict[str, Any] = Field(default=None)
    type_: str = Field(alias='type')


class FieldAssociationsRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    associationContexts: list[AssociationContextObject]
    fields: list[FieldIdentifierObject]


class AutoEnum8(_HtmlReprMixin, str, Enum):
    CUSTOM = 'custom'
    SYSTEM = 'system'


class AutoEnum9(_HtmlReprMixin, str, Enum):
    CONTEXTSCOUNT = 'contextsCount'
    CONTEXTSCOUNT_1 = '-contextsCount'
    CONTEXTSCOUNT_2 = '+contextsCount'
    LASTUSED = 'lastUsed'
    LASTUSED_1 = '-lastUsed'
    LASTUSED_2 = '+lastUsed'
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'
    SCREENSCOUNT = 'screensCount'
    SCREENSCOUNT_1 = '-screensCount'
    SCREENSCOUNT_2 = '+screensCount'
    PROJECTSCOUNT = 'projectsCount'
    PROJECTSCOUNT_1 = '-projectsCount'
    PROJECTSCOUNT_2 = '+projectsCount'


class FieldLastUsedType(_HtmlReprMixin, str, Enum):
    TRACKED = 'TRACKED'
    NOTTRACKED = 'NOT_TRACKED'
    NOINFORMATION = 'NO_INFORMATION'


class FieldLastUsed(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    type_: FieldLastUsedType = Field(default=None, alias='type')
    value: datetime = Field(default=None)


class Field_(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contextsCount: int = Field(default=None)
    description: str = Field(default=None)
    id: str
    isLocked: bool = Field(default=None)
    isUnscreenable: bool = Field(default=None)
    key: str = Field(default=None)
    lastUsed: FieldLastUsed = Field(default=None)
    name: str
    projectsCount: int = Field(default=None)
    schema: JsonTypeBean
    screensCount: int = Field(default=None)
    searcherKey: str = Field(default=None)
    stableId: str = Field(default=None)
    typeDisplayName: str = Field(default=None)


class PageBeanField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Field_] = Field(default=None)


class AutoEnum10(_HtmlReprMixin, str, Enum):
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'
    TRASHDATE = 'trashDate'
    TRASHDATE_1 = '-trashDate'
    TRASHDATE_2 = '+trashDate'
    PLANNEDDELETIONDATE = 'plannedDeletionDate'
    PLANNEDDELETIONDATE_1 = '-plannedDeletionDate'
    PLANNEDDELETIONDATE_2 = '+plannedDeletionDate'
    PROJECTSCOUNT = 'projectsCount'
    PROJECTSCOUNT_1 = '-projectsCount'
    PROJECTSCOUNT_2 = '+projectsCount'


class UpdateCustomFieldDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)
    searcherKey: CustomFieldDefinitionJsonBeanSearcherKey = Field(default=None)


class FieldProjectAssociation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectId: str = Field(default=None)


class PageBeanFieldProjectAssociation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FieldProjectAssociation] = Field(default=None)


class CustomFieldContext(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str
    id: str
    isAnyIssueType: bool
    isGlobalContext: bool
    name: str


class PageBeanCustomFieldContext(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[CustomFieldContext] = Field(default=None)


class CreateCustomFieldContext(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    issueTypeIds: list[str] = Field(default=None)
    name: str
    projectIds: list[str] = Field(default=None)


class PageBeanCustomFieldContextDefaultValue(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[
        CustomFieldContextDefaultValueCascadingOption
        | CustomFieldContextDefaultValueMultipleOption
        | CustomFieldContextDefaultValueSingleOption
        | CustomFieldContextSingleUserPickerDefaults
        | CustomFieldContextDefaultValueMultiUserPicker
        | CustomFieldContextDefaultValueSingleGroupPicker
        | CustomFieldContextDefaultValueMultipleGroupPicker
        | CustomFieldContextDefaultValueDate
        | CustomFieldContextDefaultValueDateTime
        | CustomFieldContextDefaultValueURL
        | CustomFieldContextDefaultValueProject
        | CustomFieldContextDefaultValueFloat
        | CustomFieldContextDefaultValueLabels
        | CustomFieldContextDefaultValueTextField
        | CustomFieldContextDefaultValueTextArea
        | CustomFieldContextDefaultValueReadOnly
        | CustomFieldContextDefaultValueSingleVersionPicker
        | CustomFieldContextDefaultValueMultipleVersionPicker
        | CustomFieldContextDefaultValueForgeStringField
        | CustomFieldContextDefaultValueForgeMultiStringField
        | CustomFieldContextDefaultValueForgeObjectField
        | CustomFieldContextDefaultValueForgeDateTimeField
        | CustomFieldContextDefaultValueForgeGroupField
        | CustomFieldContextDefaultValueForgeMultiGroupField
        | CustomFieldContextDefaultValueForgeNumberField
        | CustomFieldContextDefaultValueForgeUserField
        | CustomFieldContextDefaultValueForgeMultiUserField
    ] = Field(default=None)


class CustomFieldContextDefaultValueCascadingOption(_HtmlReprMixin, BaseModel):
    cascadingOptionId: str = Field(default=None)
    contextId: str
    optionId: str
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueMultipleOption(_HtmlReprMixin, BaseModel):
    contextId: str
    optionIds: list[str]
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueSingleOption(_HtmlReprMixin, BaseModel):
    contextId: str
    optionId: str
    type_: str = Field(alias='type')


class UserFilter(_HtmlReprMixin, BaseModel):
    enabled: bool
    groups: list[str] = Field(default=None)
    roleIds: list[int] = Field(default=None)


class CustomFieldContextSingleUserPickerDefaults(_HtmlReprMixin, BaseModel):
    accountId: str
    contextId: str
    type_: str = Field(alias='type')
    userFilter: UserFilter


class CustomFieldContextDefaultValueMultiUserPicker(_HtmlReprMixin, BaseModel):
    accountIds: list[str]
    contextId: str
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueSingleGroupPicker(_HtmlReprMixin, BaseModel):
    contextId: str
    groupId: str
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueMultipleGroupPicker(_HtmlReprMixin, BaseModel):
    contextId: str
    groupIds: list[str]
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueDate(_HtmlReprMixin, BaseModel):
    contextId: str
    date: str = Field(default=None)
    type_: str = Field(alias='type')
    useCurrent: bool = Field(default=False)


class CustomFieldContextDefaultValueDateTime(_HtmlReprMixin, BaseModel):
    contextId: str
    dateTime: str = Field(default=None)
    type_: str = Field(alias='type')
    useCurrent: bool = Field(default=False)


class CustomFieldContextDefaultValueURL(_HtmlReprMixin, BaseModel):
    contextId: str
    type_: str = Field(alias='type')
    url: str


class CustomFieldContextDefaultValueProject(_HtmlReprMixin, BaseModel):
    contextId: str
    projectId: str
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueFloat(_HtmlReprMixin, BaseModel):
    contextId: str
    number: float
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueLabels(_HtmlReprMixin, BaseModel):
    contextId: str
    labels: list[str]
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueTextField(_HtmlReprMixin, BaseModel):
    contextId: str
    text: str = Field(default=None)
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueTextArea(_HtmlReprMixin, BaseModel):
    contextId: str
    text: str = Field(default=None)
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueReadOnly(_HtmlReprMixin, BaseModel):
    contextId: str
    text: str = Field(default=None)
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueSingleVersionPicker(_HtmlReprMixin, BaseModel):
    contextId: str
    type_: str = Field(alias='type')
    versionId: str
    versionOrder: str = Field(default=None)


class CustomFieldContextDefaultValueMultipleVersionPicker(_HtmlReprMixin, BaseModel):
    contextId: str
    type_: str = Field(alias='type')
    versionIds: list[str]
    versionOrder: str = Field(default=None)


class CustomFieldContextDefaultValueForgeStringField(_HtmlReprMixin, BaseModel):
    contextId: str
    text: str = Field(default=None)
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueForgeMultiStringField(_HtmlReprMixin, BaseModel):
    contextId: str
    type_: str = Field(alias='type')
    values: list[str] = Field(default=None)


class CustomFieldContextDefaultValueForgeObjectField(_HtmlReprMixin, BaseModel):
    contextId: str
    object_: dict[str, Any] = Field(default=None, alias='object')
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueForgeDateTimeField(_HtmlReprMixin, BaseModel):
    contextId: str
    dateTime: str = Field(default=None)
    type_: str = Field(alias='type')
    useCurrent: bool = Field(default=False)


class CustomFieldContextDefaultValueForgeGroupField(_HtmlReprMixin, BaseModel):
    contextId: str
    groupId: str
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueForgeMultiGroupField(_HtmlReprMixin, BaseModel):
    contextId: str
    groupIds: list[str]
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueForgeNumberField(_HtmlReprMixin, BaseModel):
    contextId: str
    number: float
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueForgeUserField(_HtmlReprMixin, BaseModel):
    accountId: str
    contextId: str
    type_: str = Field(alias='type')
    userFilter: UserFilter


class CustomFieldContextDefaultValueForgeMultiUserField(_HtmlReprMixin, BaseModel):
    accountIds: list[str]
    contextId: str
    type_: str = Field(alias='type')


class CustomFieldContextDefaultValueUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultValues: list[
        CustomFieldContextDefaultValueCascadingOption
        | CustomFieldContextDefaultValueMultipleOption
        | CustomFieldContextDefaultValueSingleOption
        | CustomFieldContextSingleUserPickerDefaults
        | CustomFieldContextDefaultValueMultiUserPicker
        | CustomFieldContextDefaultValueSingleGroupPicker
        | CustomFieldContextDefaultValueMultipleGroupPicker
        | CustomFieldContextDefaultValueDate
        | CustomFieldContextDefaultValueDateTime
        | CustomFieldContextDefaultValueURL
        | CustomFieldContextDefaultValueProject
        | CustomFieldContextDefaultValueFloat
        | CustomFieldContextDefaultValueLabels
        | CustomFieldContextDefaultValueTextField
        | CustomFieldContextDefaultValueTextArea
        | CustomFieldContextDefaultValueReadOnly
        | CustomFieldContextDefaultValueSingleVersionPicker
        | CustomFieldContextDefaultValueMultipleVersionPicker
        | CustomFieldContextDefaultValueForgeStringField
        | CustomFieldContextDefaultValueForgeMultiStringField
        | CustomFieldContextDefaultValueForgeObjectField
        | CustomFieldContextDefaultValueForgeDateTimeField
        | CustomFieldContextDefaultValueForgeGroupField
        | CustomFieldContextDefaultValueForgeMultiGroupField
        | CustomFieldContextDefaultValueForgeNumberField
        | CustomFieldContextDefaultValueForgeUserField
        | CustomFieldContextDefaultValueForgeMultiUserField
    ] = Field(default=None)


class IssueTypeToContextMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contextId: str
    isAnyIssueType: bool = Field(default=None)
    issueTypeId: str = Field(default=None)


class PageBeanIssueTypeToContextMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueTypeToContextMapping] = Field(default=None)


class ProjectIssueTypeMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str
    projectId: str


class ProjectIssueTypeMappings(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    mappings: list[ProjectIssueTypeMapping]


class ContextForProjectAndIssueType(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contextId: str
    issueTypeId: str
    projectId: str


class PageBeanContextForProjectAndIssueType(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ContextForProjectAndIssueType] = Field(default=None)


class CustomFieldContextProjectMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contextId: str
    isGlobalContext: bool = Field(default=None)
    projectId: str = Field(default=None)


class PageBeanCustomFieldContextProjectMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[CustomFieldContextProjectMapping] = Field(default=None)


class CustomFieldContextUpdateDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)


class IssueTypeIds(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeIds: list[str]


class CustomFieldContextOption(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    disabled: bool
    id: str
    optionId: str = Field(default=None)
    value: str


class PageBeanCustomFieldContextOption(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[CustomFieldContextOption] = Field(default=None)


class CustomFieldOptionCreate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    disabled: bool = Field(default=None)
    optionId: str = Field(default=None)
    value: str


class BulkCustomFieldOptionCreateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    options: list[CustomFieldOptionCreate] = Field(default=None)


class CustomFieldCreatedContextOptionsList(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    options: list[CustomFieldContextOption] = Field(default=None)


class CustomFieldOptionUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    disabled: bool = Field(default=None)
    id: str
    value: str = Field(default=None)


class BulkCustomFieldOptionUpdateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    options: list[CustomFieldOptionUpdate] = Field(default=None)


class CustomFieldUpdatedContextOptionsList(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    options: list[CustomFieldOptionUpdate] = Field(default=None)


class OrderOfCustomFieldOptionsPosition(_HtmlReprMixin, str, Enum):
    FIRST = 'First'
    LAST = 'Last'


class OrderOfCustomFieldOptions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    after: str = Field(default=None)
    customFieldOptionIds: list[str]
    position: OrderOfCustomFieldOptionsPosition = Field(default=None)


class SimpleErrorCollection(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errorMessages: list[str] = Field(default=None)
    errors: dict[str, str] = Field(default=None)
    httpStatusCode: int = Field(default=None)


class TaskProgressBeanRemoveOptionFromIssuesResultUnnamedModel(SimpleErrorCollection):
    pass


class RemoveOptionFromIssuesResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errors: TaskProgressBeanRemoveOptionFromIssuesResultUnnamedModel = Field(
        default=None
    )
    modifiedIssues: list[int] = Field(default=None)
    unmodifiedIssues: list[int] = Field(default=None)


class TaskProgressBeanRemoveOptionFromIssuesResultUnnamedModel1(
    RemoveOptionFromIssuesResult
):
    pass


class TaskProgressBeanRemoveOptionFromIssuesResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    elapsedRuntime: int
    finished: int = Field(default=None)
    id: str
    lastUpdate: int
    message: str = Field(default=None)
    progress: int
    result: TaskProgressBeanRemoveOptionFromIssuesResultUnnamedModel1 = Field(
        default=None
    )
    self: str
    started: int = Field(default=None)
    status: BulkOperationProgressStatus
    submitted: int
    submittedBy: int


class ProjectIds(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectIds: list[str]


class ContextUnnamedModel(AvatarUrlsBean):
    pass


class ContextUnnamedModel1(UpdatedProjectCategory):
    pass


class ContextProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: ContextUnnamedModel = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: ContextUnnamedModel1 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class ContextUnnamedModel2(ContextProjectDetails):
    pass


class ContextScope(_HtmlReprMixin, BaseModel):
    project: ContextUnnamedModel2 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class ContextUnnamedModel3(ContextScope):
    pass


class Context(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(default=None)
    name: str = Field(default=None)
    scope: ContextUnnamedModel3 = Field(default=None)


class PageBeanContext(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Context] = Field(default=None)


class ScreenWithTabUnnamedModel(AvatarUrlsBean):
    pass


class ScreenWithTabUnnamedModel1(UpdatedProjectCategory):
    pass


class ScreenWithTabProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: ScreenWithTabUnnamedModel = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: ScreenWithTabUnnamedModel1 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class ScreenWithTabUnnamedModel2(ScreenWithTabProjectDetails):
    pass


class ScreenWithTabScope(_HtmlReprMixin, BaseModel):
    project: ScreenWithTabUnnamedModel2 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class ScreenWithTabUnnamedModel3(ScreenWithTabScope):
    pass


class ScreenableTab(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(default=None)
    name: str


class ScreenWithTabUnnamedModel4(ScreenableTab):
    pass


class ScreenWithTab(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)
    scope: ScreenWithTabUnnamedModel3 = Field(default=None)
    tab: ScreenWithTabUnnamedModel4 = Field(default=None)


class PageBeanScreenWithTab(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ScreenWithTab] = Field(default=None)


class IssueFieldOptionConfigurationEnum(_HtmlReprMixin, str, Enum):
    NOTSELECTABLE = 'notSelectable'
    DEFAULTVALUE = 'defaultValue'


class ProjectScopeBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    attributes: list[IssueFieldOptionConfigurationEnum] = Field(default=None)
    id: int = Field(default=None)


class GlobalScopeBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    attributes: list[IssueFieldOptionConfigurationEnum] = Field(default=None)


class IssueFieldOptionConfigurationUnnamedModel(GlobalScopeBean):
    pass


class IssueFieldOptionScopeBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    global_: IssueFieldOptionConfigurationUnnamedModel = Field(
        default=None, alias='global'
    )
    projects: list[int] = Field(default=None)
    projects2: list[ProjectScopeBean] = Field(default=None)


class IssueFieldOptionConfigurationUnnamedModel1(IssueFieldOptionScopeBean):
    pass


class IssueFieldOptionConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    attributes: list[IssueFieldOptionConfigurationEnum] = Field(default=None)
    scope: IssueFieldOptionConfigurationUnnamedModel1 = Field(default=None)


class IssueFieldOption(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    config: IssueFieldOptionConfiguration = Field(default=None)
    id: int
    properties: dict[str, dict[str, Any]] = Field(default=None)
    value: str


class PageBeanIssueFieldOption(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueFieldOption] = Field(default=None)


class IssueFieldOptionCreateBean(_HtmlReprMixin, BaseModel):
    config: IssueFieldOptionConfiguration = Field(default=None)
    properties: dict[str, dict[str, Any]] = Field(default=None)
    value: str


class TaskProgressBeanObject(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    elapsedRuntime: int
    finished: int = Field(default=None)
    id: str
    lastUpdate: int
    message: str = Field(default=None)
    progress: int
    result: dict[str, Any] = Field(default=None)
    self: str
    started: int = Field(default=None)
    status: BulkOperationProgressStatus
    submitted: int
    submittedBy: int


class FieldConfigurationDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None, max_length=255)
    name: str = Field(max_length=255)


class PageBeanFieldConfigurationDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FieldConfigurationDetails] = Field(default=None)


class FieldConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str
    id: int
    isDefault: bool = Field(default=None)
    name: str


class FieldConfigurationItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str
    isHidden: bool = Field(default=None)
    isRequired: bool = Field(default=None)
    renderer: str = Field(default=None)


class PageBeanFieldConfigurationItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FieldConfigurationItem] = Field(default=None)


class FieldConfigurationItemsDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldConfigurationItems: list[FieldConfigurationItem]


class FieldConfigurationScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str
    name: str


class PageBeanFieldConfigurationScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FieldConfigurationScheme] = Field(default=None)


class UpdateFieldConfigurationSchemeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None, max_length=1024)
    name: str = Field(max_length=255)


class FieldConfigurationIssueTypeItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldConfigurationId: str
    fieldConfigurationSchemeId: str
    issueTypeId: str


class PageBeanFieldConfigurationIssueTypeItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FieldConfigurationIssueTypeItem] = Field(default=None)


class FieldConfigurationSchemeProjects(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldConfigurationScheme: FieldConfigurationScheme = Field(default=None)
    projectIds: list[str]


class PageBeanFieldConfigurationSchemeProjects(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FieldConfigurationSchemeProjects] = Field(default=None)


class FieldConfigurationSchemeProjectAssociation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldConfigurationSchemeId: str = Field(default=None)
    projectId: str


class FieldConfigurationToIssueTypeMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldConfigurationId: str
    issueTypeId: str


class AssociateFieldConfigurationsWithIssueTypesRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    mappings: list[FieldConfigurationToIssueTypeMapping]


class IssueTypeIdsToRemove(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeIds: list[str]


class UserList(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    end_index: int = Field(default=None, alias='end-index')
    items: list[User] = Field(default=None)
    max_results: int = Field(default=None, alias='max-results')
    size: int = Field(default=None)
    start_index: int = Field(default=None, alias='start-index')


class FilterSubscriptionUnnamedModel1(AvatarUrlsBean):
    pass


class FilterSubscriptionUnnamedModel2(SimpleListWrapperGroupName):
    pass


class FilterSubscriptionUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: FilterSubscriptionUnnamedModel1 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: FilterSubscriptionUnnamedModel2 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class FilterSubscriptionUnnamedModel3(FilterSubscriptionUser):
    pass


class FilterSubscriptionUnnamedModel(GroupName):
    pass


class FilterSubscription(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    group: FilterSubscriptionUnnamedModel = Field(default=None)
    id: int = Field(default=None)
    user: FilterSubscriptionUnnamedModel3 = Field(default=None)


class FilterSubscriptionsList(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    end_index: int = Field(default=None, alias='end-index')
    items: list[FilterSubscription] = Field(default=None)
    max_results: int = Field(default=None, alias='max-results')
    size: int = Field(default=None)
    start_index: int = Field(default=None, alias='start-index')


class FilterUnnamedModel4(FilterSubscriptionsList):
    pass


class FilterUnnamedModel3(UserList):
    pass


class FilterUnnamedModel1(SimpleListWrapperGroupName):
    pass


class FilterUnnamedModel(AvatarUrlsBean):
    pass


class FilterUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: FilterUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: FilterUnnamedModel1 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class FilterUnnamedModel2(FilterUser):
    pass


class Filter(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    approximateLastUsed: datetime = Field(default=None)
    description: str = Field(default=None)
    editPermissions: list[SharePermission] = Field(default=None)
    favourite: bool = Field(default=None)
    favouritedCount: int = Field(default=None)
    id: str = Field(default=None)
    jql: str = Field(default=None)
    name: str
    owner: FilterUnnamedModel2 = Field(default=None)
    searchUrl: str = Field(default=None)
    self: str = Field(default=None)
    sharePermissions: list[SharePermission] = Field(default=None)
    sharedUsers: FilterUnnamedModel3 = Field(default=None)
    subscriptions: FilterUnnamedModel4 = Field(default=None)
    viewUrl: str = Field(default=None)


class DefaultShareScopeScope(_HtmlReprMixin, str, Enum):
    GLOBAL = 'GLOBAL'
    AUTHENTICATED = 'AUTHENTICATED'
    PRIVATE = 'PRIVATE'


class DefaultShareScope(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    scope: DefaultShareScopeScope


class AutoEnum11(_HtmlReprMixin, str, Enum):
    DESCRIPTION = 'description'
    DESCRIPTION_1 = '-description'
    DESCRIPTION_2 = '+description'
    FAVOURITECOUNT = 'favourite_count'
    FAVOURITECOUNT_1 = '-favourite_count'
    FAVOURITECOUNT_2 = '+favourite_count'
    ID = 'id'
    ID_1 = '-id'
    ID_2 = '+id'
    ISFAVOURITE = 'is_favourite'
    ISFAVOURITE_1 = '-is_favourite'
    ISFAVOURITE_2 = '+is_favourite'
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'
    OWNER = 'owner'
    OWNER_1 = '-owner'
    OWNER_2 = '+owner'
    ISSHARED = 'is_shared'
    ISSHARED_1 = '-is_shared'
    ISSHARED_2 = '+is_shared'


class FilterDetailsUnnamedModel1(SimpleListWrapperGroupName):
    pass


class FilterDetailsUnnamedModel(AvatarUrlsBean):
    pass


class FilterDetailsUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: FilterDetailsUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: FilterDetailsUnnamedModel1 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class FilterDetailsUnnamedModel2(FilterDetailsUser):
    pass


class FilterDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    approximateLastUsed: datetime = Field(default=None)
    description: str = Field(default=None)
    editPermissions: list[SharePermission] = Field(default=None)
    expand: str = Field(default=None)
    favourite: bool = Field(default=None)
    favouritedCount: int = Field(default=None)
    id: str = Field(default=None)
    jql: str = Field(default=None)
    name: str
    owner: FilterDetailsUnnamedModel2 = Field(default=None)
    searchUrl: str = Field(default=None)
    self: str = Field(default=None)
    sharePermissions: list[SharePermission] = Field(default=None)
    subscriptions: list[FilterSubscription] = Field(default=None)
    viewUrl: str = Field(default=None)


class PageBeanFilterDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[FilterDetails] = Field(default=None)


class ColumnItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    label: str = Field(default=None)
    value: str = Field(default=None)


class ColumnRequestBody(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    columns: list[str] = Field(default=None)


class ChangeFilterOwner(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str


class SharePermissionInputBeanType(_HtmlReprMixin, str, Enum):
    USER = 'user'
    PROJECT = 'project'
    GROUP = 'group'
    PROJECTROLE = 'projectRole'
    GLOBAL = 'global'
    AUTHENTICATED = 'authenticated'


class SharePermissionInputBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None)
    groupId: str = Field(default=None)
    groupname: str = Field(default=None)
    projectId: str = Field(default=None)
    projectRoleId: str = Field(default=None)
    rights: int = Field(default=None)
    type_: SharePermissionInputBeanType = Field(alias='type')


class ProjectPinActionAction(_HtmlReprMixin, str, Enum):
    PIN = 'PIN'
    UNPIN = 'UNPIN'


class ProjectPinAction(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    action: ProjectPinActionAction
    projectIdOrKey: str


class ForgePanelProjectPinRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    moduleId: str
    projectList: list[ProjectPinAction] = Field(max_length=1000)


class ForgePanelProjectPinAsyncResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    taskId: str = Field(default=None)


class PagedListUserDetailsApplicationUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    end_index: int = Field(default=None, alias='end-index')
    items: list[UserDetails] = Field(default=None)
    max_results: int = Field(default=None, alias='max-results')
    size: int = Field(default=None)
    start_index: int = Field(default=None, alias='start-index')


class GroupUnnamedModel(PagedListUserDetailsApplicationUser):
    pass


class Group(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: str = Field(default=None)
    groupId: str | None = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    users: GroupUnnamedModel = Field(default=None)


class AddGroupBean(_HtmlReprMixin, BaseModel):
    name: str


class GroupDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    groupId: str | None = Field(default=None)
    name: str = Field(default=None)


class PageBeanGroupDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[GroupDetails] = Field(default=None)


class PageBeanUserDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[UserDetails] = Field(default=None)


class UpdateUserToGroupBean(_HtmlReprMixin, BaseModel):
    accountId: str = Field(default=None, max_length=128)
    name: str = Field(default=None)


class FoundGroupManagedBy(_HtmlReprMixin, str, Enum):
    EXTERNAL = 'EXTERNAL'
    ADMINS = 'ADMINS'
    TEAMMEMBERS = 'TEAM_MEMBERS'
    OPEN = 'OPEN'


class GroupLabelType(_HtmlReprMixin, str, Enum):
    ADMIN = 'ADMIN'
    SINGLE = 'SINGLE'
    MULTIPLE = 'MULTIPLE'


class GroupLabel(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    text: str = Field(default=None)
    title: str = Field(default=None)
    type_: GroupLabelType = Field(default=None, alias='type')


class FoundGroupUsageType(_HtmlReprMixin, str, Enum):
    USERBASEGROUP = 'USERBASE_GROUP'
    TEAMCOLLABORATION = 'TEAM_COLLABORATION'
    ADMINOVERSIGHT = 'ADMIN_OVERSIGHT'


class FoundGroup(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrl: str = Field(default=None)
    groupId: str = Field(default=None)
    html: str = Field(default=None)
    labels: list[GroupLabel] = Field(default=None)
    managedBy: FoundGroupManagedBy = Field(default=None)
    name: str = Field(default=None)
    usageType: FoundGroupUsageType = Field(default=None)


class FoundGroups(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    groups: list[FoundGroup] = Field(default=None)
    header: str = Field(default=None)
    total: int = Field(default=None)


class AutoEnum12(_HtmlReprMixin, str, Enum):
    XSMALL = 'xsmall'
    XSMALL2X = 'xsmall@2x'
    XSMALL3X = 'xsmall@3x'
    SMALL = 'small'
    SMALL2X = 'small@2x'
    SMALL3X = 'small@3x'
    MEDIUM = 'medium'
    MEDIUM2X = 'medium@2x'
    MEDIUM3X = 'medium@3x'
    LARGE = 'large'
    LARGE2X = 'large@2x'
    LARGE3X = 'large@3x'
    XLARGE = 'xlarge'
    XLARGE2X = 'xlarge@2x'
    XLARGE3X = 'xlarge@3x'
    XXLARGE = 'xxlarge'
    XXLARGE2X = 'xxlarge@2x'
    XXLARGE3X = 'xxlarge@3x'
    XXXLARGE = 'xxxlarge'
    XXXLARGE2X = 'xxxlarge@2x'
    XXXLARGE3X = 'xxxlarge@3x'


class UserPickerUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    avatarUrl: str = Field(default=None)
    displayName: str = Field(default=None)
    html: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)


class FoundUsers(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    header: str = Field(default=None)
    total: int = Field(default=None)
    users: list[UserPickerUser] = Field(default=None)


class FoundUsersAndGroups(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    groups: FoundGroups = Field(default=None)
    users: FoundUsers = Field(default=None)


class LicensedApplicationPlan(_HtmlReprMixin, str, Enum):
    UNLICENSED = 'UNLICENSED'
    FREE = 'FREE'
    PAID = 'PAID'


class LicensedApplication(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str
    plan: LicensedApplicationPlan


class License(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    applications: list[LicensedApplication]


class IssueUpdateDetailsUnnamedModel5(UpdatedProjectCategory):
    pass


class IssueUpdateDetailsUnnamedModel4(AvatarUrlsBean):
    pass


class IssueUpdateDetailsProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: IssueUpdateDetailsUnnamedModel4 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: IssueUpdateDetailsUnnamedModel5 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class IssueUpdateDetailsUnnamedModel6(IssueUpdateDetailsProjectDetails):
    pass


class IssueUpdateDetailsScope(_HtmlReprMixin, BaseModel):
    project: IssueUpdateDetailsUnnamedModel6 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class StatusCategory(_HtmlReprMixin, BaseModel):
    colorName: str = Field(default=None)
    id: int = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class IssueUpdateDetailsUnnamedModel8(StatusCategory):
    pass


class IssueUpdateDetailsUnnamedModel1(HistoryMetadataParticipant):
    pass


class IssueUpdateDetailsUnnamedModel2(HistoryMetadataParticipant):
    pass


class IssueUpdateDetailsUnnamedModel(HistoryMetadataParticipant):
    pass


class IssueUpdateDetailsHistoryMetadata(_HtmlReprMixin, BaseModel):
    activityDescription: str = Field(default=None)
    activityDescriptionKey: str = Field(default=None)
    actor: IssueUpdateDetailsUnnamedModel = Field(default=None)
    cause: IssueUpdateDetailsUnnamedModel1 = Field(default=None)
    description: str = Field(default=None)
    descriptionKey: str = Field(default=None)
    emailDescription: str = Field(default=None)
    emailDescriptionKey: str = Field(default=None)
    extraData: dict[str, str] = Field(default=None)
    generator: IssueUpdateDetailsUnnamedModel2 = Field(default=None)
    type_: str = Field(default=None, alias='type')


class IssueUpdateDetailsUnnamedModel7(IssueUpdateDetailsScope):
    pass


class StatusDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: IssueUpdateDetailsUnnamedModel7 = Field(default=None)
    self: str = Field(default=None)
    statusCategory: IssueUpdateDetailsUnnamedModel8 = Field(default=None)


class IssueUpdateDetailsUnnamedModel9(StatusDetails):
    pass


class FieldMetadataUnnamedModel(JsonTypeBean):
    pass


class FieldMetadata(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    allowedValues: list[dict[str, Any]] = Field(default=None)
    autoCompleteUrl: str = Field(default=None)
    configuration: dict[str, dict[str, Any]] = Field(default=None)
    defaultValue: dict[str, Any] = Field(default=None)
    hasDefaultValue: bool = Field(default=None)
    key: str
    name: str
    operations: list[str]
    required: bool
    schema: FieldMetadataUnnamedModel


class IssueUpdateDetailsUnnamedModel3(IssueUpdateDetailsHistoryMetadata):
    pass


class IssueTransition(_HtmlReprMixin, BaseModel):
    expand: str = Field(default=None)
    fields: dict[str, FieldMetadata] = Field(default=None)
    hasScreen: bool = Field(default=None)
    id: str = Field(default=None)
    isAvailable: bool = Field(default=None)
    isConditional: bool = Field(default=None)
    isGlobal: bool = Field(default=None)
    isInitial: bool = Field(default=None)
    looped: bool = Field(default=None)
    name: str = Field(default=None)
    to: IssueUpdateDetailsUnnamedModel9 = Field(default=None)


class FieldUpdateOperation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    add: dict[str, Any] = Field(default=None)
    copy: dict[str, Any] = Field(default=None)
    edit: dict[str, Any] = Field(default=None)
    remove: dict[str, Any] = Field(default=None)
    set_: dict[str, Any] = Field(default=None, alias='set')


class IssueUpdateDetailsUnnamedModel10(IssueTransition):
    pass


class IssueUpdateDetails(_HtmlReprMixin, BaseModel):
    fields: dict[str, dict[str, Any]] = Field(default=None)
    historyMetadata: IssueUpdateDetailsUnnamedModel3 = Field(default=None)
    properties: list[EntityProperty] = Field(default=None)
    transition: IssueUpdateDetailsUnnamedModel10 = Field(default=None)
    update: dict[str, list[FieldUpdateOperation]] = Field(default=None)


class WarningCollection(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    warnings: list[str] = Field(default=None)


class NestedResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errorCollection: ErrorCollection = Field(default=None)
    status: int = Field(default=None)
    warningCollection: WarningCollection = Field(default=None)


class CreatedIssueUnnamedModel(NestedResponse):
    pass


class CreatedIssueUnnamedModel1(NestedResponse):
    pass


class CreatedIssue(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    key: str = Field(default=None)
    self: str = Field(default=None)
    transition: CreatedIssueUnnamedModel = Field(default=None)
    watchers: CreatedIssueUnnamedModel1 = Field(default=None)


class ArchiveIssueAsyncRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    jql: str = Field(default=None)


class IssueArchivalSyncRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueIdsOrKeys: list[str] = Field(default=None)


class Error(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    count: int = Field(default=None)
    issueIdsOrKeys: list[str] = Field(default=None)
    message: str = Field(default=None)


class Errors(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueIsSubtask: Error = Field(default=None)
    issuesInArchivedProjects: Error = Field(default=None)
    issuesInUnlicensedProjects: Error = Field(default=None)
    issuesNotFound: Error = Field(default=None)
    userDoesNotHavePermission: Error = Field(default=None)


class IssueArchivalSyncResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errors: Errors = Field(default=None)
    numberOfIssuesUpdated: int = Field(default=None)


class IssuesUpdateBean(_HtmlReprMixin, BaseModel):
    issueUpdates: list[IssueUpdateDetails] = Field(default=None)


class BulkOperationErrorResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    elementErrors: ErrorCollection = Field(default=None)
    failedElementNumber: int = Field(default=None)
    status: int = Field(default=None)


class CreatedIssues(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errors: list[BulkOperationErrorResult] = Field(default=None)
    issues: list[CreatedIssue] = Field(default=None)


class BulkFetchIssueRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: list[str] = Field(default=None)
    fields: list[str] = Field(default=None)
    fieldsByKeys: bool = Field(default=None)
    issueIdsOrKeys: list[str]
    properties: list[str] = Field(default=None)


class IncludedFields(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    actuallyIncluded: list[str] = Field(default=None)
    excluded: list[str] = Field(default=None)
    included: list[str] = Field(default=None)


class LinkGroup(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    groups: list[LinkGroup] = Field(default=None)
    header: SimpleLink = Field(default=None)
    id: str = Field(default=None)
    links: list[SimpleLink] = Field(default=None)
    styleClass: str = Field(default=None)
    weight: int = Field(default=None)


class Operations(_HtmlReprMixin, BaseModel):
    linkGroups: list[LinkGroup] = Field(default=None)


class PageOfChangelogs(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    histories: list[Changelog] = Field(default=None)
    maxResults: int = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)


class IssueBeanUnnamedModel(PageOfChangelogs):
    pass


class IssueUpdateMetadata(_HtmlReprMixin, BaseModel):
    fields: dict[str, FieldMetadata] = Field(default=None)


class IssueBeanUnnamedModel1(IssueUpdateMetadata):
    pass


class IssueError(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errorMessage: str = Field(default=None)
    id: str = Field(default=None)


class IssueBeanUnnamedModel2(Operations):
    pass


class IssueBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    changelog: IssueBeanUnnamedModel = Field(default=None)
    editmeta: IssueBeanUnnamedModel1 = Field(default=None)
    expand: str = Field(default=None)
    fields: dict[str, dict[str, Any]] = Field(default=None)
    fieldsToInclude: IncludedFields = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    names: dict[str, str] = Field(default=None)
    operations: IssueBeanUnnamedModel2 = Field(default=None)
    properties: dict[str, dict[str, Any]] = Field(default=None)
    renderedFields: dict[str, dict[str, Any]] = Field(default=None)
    schema: dict[str, JsonTypeBean] = Field(default=None)
    self: str = Field(default=None)
    transitions: list[IssueTransition] = Field(default=None)
    versionedRepresentations: dict[str, dict[str, dict[str, Any]]] = Field(default=None)


class BulkIssueResults(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueErrors: list[IssueError] = Field(default=None)
    issues: list[IssueBean] = Field(default=None)


class IssueTypeIssueCreateMetadataUnnamedModel1(UpdatedProjectCategory):
    pass


class IssueTypeIssueCreateMetadataUnnamedModel(AvatarUrlsBean):
    pass


class IssueTypeIssueCreateMetadataProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: IssueTypeIssueCreateMetadataUnnamedModel = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: IssueTypeIssueCreateMetadataUnnamedModel1 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class IssueTypeIssueCreateMetadataUnnamedModel2(
    IssueTypeIssueCreateMetadataProjectDetails
):
    pass


class IssueTypeIssueCreateMetadataScope(_HtmlReprMixin, BaseModel):
    project: IssueTypeIssueCreateMetadataUnnamedModel2 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class IssueTypeIssueCreateMetadataUnnamedModel3(IssueTypeIssueCreateMetadataScope):
    pass


class ProjectIssueCreateMetadataUnnamedModel(AvatarUrlsBean):
    pass


class IssueTypeIssueCreateMetadata(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    entityId: UUID = Field(default=None)
    expand: str = Field(default=None)
    fields: dict[str, FieldMetadata] = Field(default=None)
    hierarchyLevel: int = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: IssueTypeIssueCreateMetadataUnnamedModel3 = Field(default=None)
    self: str = Field(default=None)
    subtask: bool = Field(default=None)


class ProjectIssueCreateMetadata(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: ProjectIssueCreateMetadataUnnamedModel = Field(default=None)
    expand: str = Field(default=None)
    id: str = Field(default=None)
    issuetypes: list[IssueTypeIssueCreateMetadata] = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class IssueCreateMetadata(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: str = Field(default=None)
    projects: list[ProjectIssueCreateMetadata] = Field(default=None)


class PageOfCreateMetaIssueTypes(_HtmlReprMixin, BaseModel):
    createMetaIssueType: list[IssueTypeIssueCreateMetadata] = Field(default=None)
    issueTypes: list[IssueTypeIssueCreateMetadata] = Field(default=None)
    maxResults: int = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)


class FieldCreateMetadataUnnamedModel(JsonTypeBean):
    pass


class FieldCreateMetadata(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    allowedValues: list[dict[str, Any]] = Field(default=None)
    autoCompleteUrl: str = Field(default=None)
    configuration: dict[str, dict[str, Any]] = Field(default=None)
    defaultValue: dict[str, Any] = Field(default=None)
    fieldId: str
    hasDefaultValue: bool = Field(default=None)
    key: str
    name: str
    operations: list[str]
    required: bool
    schema: FieldCreateMetadataUnnamedModel


class PageOfCreateMetaIssueTypeWithField(_HtmlReprMixin, BaseModel):
    fields: list[FieldCreateMetadata] = Field(default=None)
    maxResults: int = Field(default=None)
    results: list[FieldCreateMetadata] = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)


class IssueLimitReportResponseBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issuesApproachingLimit: dict[str, dict[str, int]] = Field(default=None)
    issuesBreachingLimit: dict[str, dict[str, int]] = Field(default=None)
    limits: dict[str, int] = Field(default=None)


class SuggestedIssue(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(default=None)
    img: str = Field(default=None)
    key: str = Field(default=None)
    keyHtml: str = Field(default=None)
    summary: str = Field(default=None)
    summaryText: str = Field(default=None)


class IssuePickerSuggestionsIssueType(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    issues: list[SuggestedIssue] = Field(default=None)
    label: str = Field(default=None)
    msg: str = Field(default=None)
    sub: str = Field(default=None)


class IssuePickerSuggestions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    sections: list[IssuePickerSuggestionsIssueType] = Field(default=None)


class JsonNodeNumberType(_HtmlReprMixin, str, Enum):
    INT = 'INT'
    LONG = 'LONG'
    BIGINTEGER = 'BIG_INTEGER'
    FLOAT = 'FLOAT'
    DOUBLE = 'DOUBLE'
    BIGDECIMAL = 'BIG_DECIMAL'


class JsonNode(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    array: bool = Field(default=None)
    bigDecimal: bool = Field(default=None)
    bigInteger: bool = Field(default=None)
    bigIntegerValue: int = Field(default=None)
    binary: bool = Field(default=None)
    binaryValue: list[bytes] = Field(default=None)
    boolean: bool = Field(default=None)
    booleanValue: bool = Field(default=None)
    containerNode: bool = Field(default=None)
    decimalValue: float = Field(default=None)
    double: bool = Field(default=None)
    doubleValue: float = Field(default=None)
    elements: dict[str, Any] = Field(default=None)
    fieldNames: dict[str, Any] = Field(default=None)
    fields: dict[str, Any] = Field(default=None)
    floatingPointNumber: bool = Field(default=None)
    int_: bool = Field(default=None, alias='int')
    intValue: int = Field(default=None)
    integralNumber: bool = Field(default=None)
    long: bool = Field(default=None)
    longValue: int = Field(default=None)
    missingNode: bool = Field(default=None)
    null: bool = Field(default=None)
    number: bool = Field(default=None)
    numberType: JsonNodeNumberType = Field(default=None)
    numberValue: float = Field(default=None)
    object_: bool = Field(default=None, alias='object')
    pojo: bool = Field(default=None)
    textValue: str = Field(default=None)
    textual: bool = Field(default=None)
    valueAsBoolean: bool = Field(default=None)
    valueAsDouble: float = Field(default=None)
    valueAsInt: int = Field(default=None)
    valueAsLong: int = Field(default=None)
    valueAsText: str = Field(default=None)
    valueNode: bool = Field(default=None)


class IssueEntityProperties(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    entitiesIds: list[int] = Field(default=None, min_length=1, max_length=10000)
    properties: dict[str, JsonNode] = Field(default=None)


class IssueEntityPropertiesForMultiUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueID: int = Field(default=None)
    properties: dict[str, JsonNode] = Field(default=None)


class MultiIssueEntityProperties(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issues: list[IssueEntityPropertiesForMultiUpdate] = Field(default=None)


class IssueFilterForBulkPropertySet(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    currentValue: dict[str, Any] = Field(default=None)
    entityIds: list[int] = Field(default=None)
    hasProperty: bool = Field(default=None)


class BulkIssuePropertyUpdateRequestUnnamedModel(IssueFilterForBulkPropertySet):
    pass


class BulkIssuePropertyUpdateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expression: str = Field(default=None)
    filter: BulkIssuePropertyUpdateRequestUnnamedModel = Field(default=None)
    value: dict[str, Any] = Field(default=None)


class IssueFilterForBulkPropertyDelete(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    currentValue: dict[str, Any] = Field(default=None)
    entityIds: list[int] = Field(default=None)


class IssueList(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueIds: list[str]


class BulkIssueIsWatching(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issuesIsWatching: dict[str, bool] = Field(default=None)


class AutoEnum13(_HtmlReprMixin, str, Enum):
    TRUE = 'true'
    FALSE = 'false'


class Resource(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contentAsByteArray: list[bytes] = Field(default=None)
    description: str = Field(default=None)
    file: bytes = Field(default=None)
    filename: str = Field(default=None)
    inputStream: dict[str, Any] = Field(default=None)
    open: bool = Field(default=None)
    readable: bool = Field(default=None)
    uri: str = Field(default=None)
    url: str = Field(default=None)


class MultipartFile(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    bytes_: list[bytes] = Field(default=None, alias='bytes')
    contentType: str = Field(default=None)
    empty: bool = Field(default=None)
    inputStream: dict[str, Any] = Field(default=None)
    name: str = Field(default=None)
    originalFilename: str = Field(default=None)
    resource: Resource = Field(default=None)
    size: int = Field(default=None)


class AttachmentUnnamedModel(AvatarUrlsBean):
    pass


class AttachmentUserDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: AttachmentUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class AttachmentUnnamedModel1(AttachmentUserDetails):
    pass


class Attachment(_HtmlReprMixin, BaseModel):
    author: AttachmentUnnamedModel1 = Field(default=None)
    content: str = Field(default=None)
    created: datetime = Field(default=None)
    filename: str = Field(default=None)
    id: str = Field(default=None)
    mimeType: str = Field(default=None)
    self: str = Field(default=None)
    size: int = Field(default=None)
    thumbnail: str = Field(default=None)


class PageBeanChangelog(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Changelog] = Field(default=None)


class IssueChangelogIds(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    changelogIds: list[int]


class AutoEnum14(_HtmlReprMixin, str, Enum):
    CREATED = 'created'
    CREATED_1 = '-created'
    CREATED_2 = '+created'


class PageOfComments(_HtmlReprMixin, BaseModel):
    comments: list[Comment] = Field(default=None)
    maxResults: int = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)


class RestrictedPermission(_HtmlReprMixin, BaseModel):
    id: str = Field(default=None)
    key: str = Field(default=None)


class NotificationRecipients(_HtmlReprMixin, BaseModel):
    assignee: bool = Field(default=None)
    groupIds: list[str] = Field(default=None)
    groups: list[GroupName] = Field(default=None)
    reporter: bool = Field(default=None)
    users: list[UserDetails] = Field(default=None)
    voters: bool = Field(default=None)
    watchers: bool = Field(default=None)


class NotificationRecipientsRestrictions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    groupIds: list[str] = Field(default=None)
    groups: list[GroupName] = Field(default=None)
    permissions: list[RestrictedPermission] = Field(default=None)


class NotificationUnnamedModel1(NotificationRecipients):
    pass


class NotificationUnnamedModel(NotificationRecipientsRestrictions):
    pass


class Notification(_HtmlReprMixin, BaseModel):
    htmlBody: str = Field(default=None)
    restrict: NotificationUnnamedModel = Field(default=None)
    subject: str = Field(default=None)
    textBody: str = Field(default=None)
    to: NotificationUnnamedModel1 = Field(default=None)


class Application(_HtmlReprMixin, BaseModel):
    name: str = Field(default=None)
    type_: str = Field(default=None, alias='type')


class Icon(_HtmlReprMixin, BaseModel):
    link: str = Field(default=None)
    title: str = Field(default=None)
    url16x16: str = Field(default=None)


class RemoteIssueLinkUnnamedModel2(Icon):
    pass


class RemoteIssueLinkUnnamedModel1(Icon):
    pass


class Status(_HtmlReprMixin, BaseModel):
    icon: RemoteIssueLinkUnnamedModel2 = Field(default=None)
    resolved: bool = Field(default=None)


class RemoteIssueLinkUnnamedModel3(Status):
    pass


class RemoteObject(_HtmlReprMixin, BaseModel):
    icon: RemoteIssueLinkUnnamedModel1 = Field(default=None)
    status: RemoteIssueLinkUnnamedModel3 = Field(default=None)
    summary: str = Field(default=None)
    title: str
    url: str


class RemoteIssueLinkUnnamedModel4(RemoteObject):
    pass


class RemoteIssueLinkUnnamedModel(Application):
    pass


class RemoteIssueLink(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    application: RemoteIssueLinkUnnamedModel = Field(default=None)
    globalId: str = Field(default=None)
    id: int = Field(default=None)
    object_: RemoteIssueLinkUnnamedModel4 = Field(default=None, alias='object')
    relationship: str = Field(default=None)
    self: str = Field(default=None)


class RemoteIssueLinkRequestUnnamedModel2(Icon):
    pass


class RemoteIssueLinkRequestStatus(_HtmlReprMixin, BaseModel):
    icon: RemoteIssueLinkRequestUnnamedModel2 = Field(default=None)
    resolved: bool = Field(default=None)


class RemoteIssueLinkRequestUnnamedModel3(RemoteIssueLinkRequestStatus):
    pass


class RemoteIssueLinkRequestUnnamedModel(Application):
    pass


class RemoteIssueLinkRequestUnnamedModel1(Icon):
    pass


class RemoteIssueLinkRequestRemoteObject(_HtmlReprMixin, BaseModel):
    icon: RemoteIssueLinkRequestUnnamedModel1 = Field(default=None)
    status: RemoteIssueLinkRequestUnnamedModel3 = Field(default=None)
    summary: str = Field(default=None)
    title: str
    url: str


class RemoteIssueLinkRequestUnnamedModel4(RemoteIssueLinkRequestRemoteObject):
    pass


class RemoteIssueLinkRequest(_HtmlReprMixin, BaseModel):
    application: RemoteIssueLinkRequestUnnamedModel = Field(default=None)
    globalId: str = Field(default=None)
    object_: RemoteIssueLinkRequestUnnamedModel4 = Field(alias='object')
    relationship: str = Field(default=None)


class RemoteIssueLinkIdentifies(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(default=None)
    self: str = Field(default=None)


class Transitions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: str = Field(default=None)
    transitions: list[IssueTransition] = Field(default=None)


class Votes(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    hasVoted: bool = Field(default=None)
    self: str = Field(default=None)
    voters: list[User] = Field(default=None)
    votes: int = Field(default=None)


class Watchers(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isWatching: bool = Field(default=None)
    self: str = Field(default=None)
    watchCount: int = Field(default=None)
    watchers: list[UserDetails] = Field(default=None)


class WorklogUnnamedModel(AvatarUrlsBean):
    pass


class WorklogUserDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: WorklogUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class WorklogUnnamedModel1(WorklogUserDetails):
    pass


class WorklogUnnamedModel2(AvatarUrlsBean):
    pass


class WorklogUserDetails1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: WorklogUnnamedModel2 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class WorklogUnnamedModel4(Visibility):
    pass


class WorklogUnnamedModel3(WorklogUserDetails1):
    pass


class Worklog(_HtmlReprMixin, BaseModel):
    author: WorklogUnnamedModel1 = Field(default=None)
    comment: dict[str, Any] = Field(default=None)
    created: datetime = Field(default=None)
    id: str = Field(default=None)
    issueId: str = Field(default=None)
    properties: list[EntityProperty] = Field(default=None)
    self: str = Field(default=None)
    started: datetime = Field(default=None)
    timeSpent: str = Field(default=None)
    timeSpentSeconds: int = Field(default=None)
    updateAuthor: WorklogUnnamedModel3 = Field(default=None)
    updated: datetime = Field(default=None)
    visibility: WorklogUnnamedModel4 = Field(default=None)


class PageOfWorklogs(_HtmlReprMixin, BaseModel):
    maxResults: int = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    worklogs: list[Worklog] = Field(default=None)


class AutoEnum15(_HtmlReprMixin, str, Enum):
    NEW = 'new'
    LEAVE = 'leave'
    MANUAL = 'manual'
    AUTO = 'auto'


class AutoEnum16(_HtmlReprMixin, str, Enum):
    LEAVE = 'leave'
    AUTO = 'auto'


class WorklogIdsRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    ids: list[int]


class WorklogsMoveRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    ids: list[int] = Field(default=None)
    issueIdOrKey: str = Field(default=None)


class LinkedIssueUnnamedModel10(UpdatedProjectCategory):
    pass


class ExpandPrioritySchemePage(_HtmlReprMixin, BaseModel):
    maxResults: int = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)


class LinkedIssueUnnamedModel7(ExpandPrioritySchemePage):
    pass


class Priority(_HtmlReprMixin, BaseModel):
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    isDefault: bool = Field(default=None)
    name: str = Field(default=None)
    schemes: LinkedIssueUnnamedModel7 = Field(default=None)
    self: str = Field(default=None)
    statusColor: str = Field(default=None)


class LinkedIssueUnnamedModel8(Priority):
    pass


class TimeTrackingDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    originalEstimate: str = Field(default=None)
    originalEstimateSeconds: int = Field(default=None)
    remainingEstimate: str = Field(default=None)
    remainingEstimateSeconds: int = Field(default=None)
    timeSpent: str = Field(default=None)
    timeSpentSeconds: int = Field(default=None)


class LinkedIssueUnnamedModel3(UpdatedProjectCategory):
    pass


class LinkedIssueUnnamedModel2(AvatarUrlsBean):
    pass


class LinkedIssueProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: LinkedIssueUnnamedModel2 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: LinkedIssueUnnamedModel3 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class LinkedIssueUnnamedModel4(LinkedIssueProjectDetails):
    pass


class LinkedIssueScope(_HtmlReprMixin, BaseModel):
    project: LinkedIssueUnnamedModel4 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class LinkedIssueUnnamedModel5(LinkedIssueScope):
    pass


class LinkedIssueIssueTypeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    entityId: UUID = Field(default=None)
    hierarchyLevel: int = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: LinkedIssueUnnamedModel5 = Field(default=None)
    self: str = Field(default=None)
    subtask: bool = Field(default=None)


class LinkedIssueUnnamedModel(AvatarUrlsBean):
    pass


class LinkedIssueUnnamedModel9(AvatarUrlsBean):
    pass


class LinkedIssueProjectDetails1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: LinkedIssueUnnamedModel9 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: LinkedIssueUnnamedModel10 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class LinkedIssueUnnamedModel11(LinkedIssueProjectDetails1):
    pass


class LinkedIssueScope1(_HtmlReprMixin, BaseModel):
    project: LinkedIssueUnnamedModel11 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class LinkedIssueUnnamedModel12(LinkedIssueScope1):
    pass


class LinkedIssueUnnamedModel15(TimeTrackingDetails):
    pass


class LinkedIssueUserDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: LinkedIssueUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class LinkedIssueUnnamedModel1(LinkedIssueUserDetails):
    pass


class LinkedIssueUnnamedModel6(LinkedIssueIssueTypeDetails):
    pass


class LinkedIssueUnnamedModel13(StatusCategory):
    pass


class LinkedIssueStatusDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: LinkedIssueUnnamedModel12 = Field(default=None)
    self: str = Field(default=None)
    statusCategory: LinkedIssueUnnamedModel13 = Field(default=None)


class LinkedIssueUnnamedModel14(LinkedIssueStatusDetails):
    pass


class Fields(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    assignee: LinkedIssueUnnamedModel1 = Field(default=None)
    issueType: LinkedIssueUnnamedModel6 = Field(default=None)
    issuetype: dict[str, Any] = Field(default=None)
    priority: LinkedIssueUnnamedModel8 = Field(default=None)
    status: LinkedIssueUnnamedModel14 = Field(default=None)
    summary: str = Field(default=None)
    timetracking: LinkedIssueUnnamedModel15 = Field(default=None)


class LinkedIssueUnnamedModel16(Fields):
    pass


class LinkedIssue(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fields: LinkedIssueUnnamedModel16 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    self: str = Field(default=None)


class IssueLinkType(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    inward: str = Field(default=None)
    name: str = Field(default=None)
    outward: str = Field(default=None)
    self: str = Field(default=None)


class LinkIssueRequestJsonBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    comment: Comment = Field(default=None)
    inwardIssue: LinkedIssue
    outwardIssue: LinkedIssue
    type_: IssueLinkType = Field(alias='type')


class IssueLinkUnnamedModel28(UpdatedProjectCategory):
    pass


class IssueLinkUnnamedModel27(AvatarUrlsBean):
    pass


class IssueLinkProjectDetails3(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: IssueLinkUnnamedModel27 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: IssueLinkUnnamedModel28 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class IssueLinkUnnamedModel29(IssueLinkProjectDetails3):
    pass


class IssueLinkScope3(_HtmlReprMixin, BaseModel):
    project: IssueLinkUnnamedModel29 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class IssueLinkUnnamedModel30(IssueLinkScope3):
    pass


class IssueLinkUnnamedModel18(AvatarUrlsBean):
    pass


class IssueLinkUserDetails1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: IssueLinkUnnamedModel18 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class IssueLinkUnnamedModel19(IssueLinkUserDetails1):
    pass


class IssueLinkUnnamedModel21(UpdatedProjectCategory):
    pass


class IssueLinkUnnamedModel20(AvatarUrlsBean):
    pass


class IssueLinkProjectDetails2(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: IssueLinkUnnamedModel20 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: IssueLinkUnnamedModel21 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class IssueLinkUnnamedModel22(IssueLinkProjectDetails2):
    pass


class IssueLinkScope2(_HtmlReprMixin, BaseModel):
    project: IssueLinkUnnamedModel22 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class IssueLinkUnnamedModel23(IssueLinkScope2):
    pass


class IssueLinkIssueTypeDetails1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    entityId: UUID = Field(default=None)
    hierarchyLevel: int = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: IssueLinkUnnamedModel23 = Field(default=None)
    self: str = Field(default=None)
    subtask: bool = Field(default=None)


class IssueLinkUnnamedModel24(IssueLinkIssueTypeDetails1):
    pass


class IssueLinkUnnamedModel31(StatusCategory):
    pass


class IssueLinkUnnamedModel33(TimeTrackingDetails):
    pass


class IssueLinkStatusDetails1(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: IssueLinkUnnamedModel30 = Field(default=None)
    self: str = Field(default=None)
    statusCategory: IssueLinkUnnamedModel31 = Field(default=None)


class IssueLinkUnnamedModel32(IssueLinkStatusDetails1):
    pass


class IssueLinkUnnamedModel25(ExpandPrioritySchemePage):
    pass


class IssueLinkPriority1(_HtmlReprMixin, BaseModel):
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    isDefault: bool = Field(default=None)
    name: str = Field(default=None)
    schemes: IssueLinkUnnamedModel25 = Field(default=None)
    self: str = Field(default=None)
    statusColor: str = Field(default=None)


class IssueLinkUnnamedModel26(IssueLinkPriority1):
    pass


class IssueLinkFields1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    assignee: IssueLinkUnnamedModel19 = Field(default=None)
    issueType: IssueLinkUnnamedModel24 = Field(default=None)
    issuetype: dict[str, Any] = Field(default=None)
    priority: IssueLinkUnnamedModel26 = Field(default=None)
    status: IssueLinkUnnamedModel32 = Field(default=None)
    summary: str = Field(default=None)
    timetracking: IssueLinkUnnamedModel33 = Field(default=None)


class IssueLinkUnnamedModel7(ExpandPrioritySchemePage):
    pass


class IssueLinkPriority(_HtmlReprMixin, BaseModel):
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    isDefault: bool = Field(default=None)
    name: str = Field(default=None)
    schemes: IssueLinkUnnamedModel7 = Field(default=None)
    self: str = Field(default=None)
    statusColor: str = Field(default=None)


class IssueLinkUnnamedModel13(StatusCategory):
    pass


class IssueLinkUnnamedModel3(UpdatedProjectCategory):
    pass


class IssueLinkUnnamedModel9(AvatarUrlsBean):
    pass


class IssueLinkUnnamedModel10(UpdatedProjectCategory):
    pass


class IssueLinkProjectDetails1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: IssueLinkUnnamedModel9 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: IssueLinkUnnamedModel10 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class IssueLinkUnnamedModel11(IssueLinkProjectDetails1):
    pass


class IssueLinkScope1(_HtmlReprMixin, BaseModel):
    project: IssueLinkUnnamedModel11 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class IssueLinkUnnamedModel12(IssueLinkScope1):
    pass


class IssueLinkStatusDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: IssueLinkUnnamedModel12 = Field(default=None)
    self: str = Field(default=None)
    statusCategory: IssueLinkUnnamedModel13 = Field(default=None)


class IssueLinkUnnamedModel14(IssueLinkStatusDetails):
    pass


class IssueLinkUnnamedModel2(AvatarUrlsBean):
    pass


class IssueLinkProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: IssueLinkUnnamedModel2 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: IssueLinkUnnamedModel3 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class IssueLinkUnnamedModel4(IssueLinkProjectDetails):
    pass


class IssueLinkScope(_HtmlReprMixin, BaseModel):
    project: IssueLinkUnnamedModel4 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class IssueLinkUnnamedModel5(IssueLinkScope):
    pass


class IssueLinkIssueTypeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    entityId: UUID = Field(default=None)
    hierarchyLevel: int = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: IssueLinkUnnamedModel5 = Field(default=None)
    self: str = Field(default=None)
    subtask: bool = Field(default=None)


class IssueLinkUnnamedModel(AvatarUrlsBean):
    pass


class IssueLinkUserDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: IssueLinkUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class IssueLinkUnnamedModel1(IssueLinkUserDetails):
    pass


class IssueLinkUnnamedModel6(IssueLinkIssueTypeDetails):
    pass


class IssueLinkUnnamedModel8(IssueLinkPriority):
    pass


class IssueLinkUnnamedModel15(TimeTrackingDetails):
    pass


class IssueLinkFields(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    assignee: IssueLinkUnnamedModel1 = Field(default=None)
    issueType: IssueLinkUnnamedModel6 = Field(default=None)
    issuetype: dict[str, Any] = Field(default=None)
    priority: IssueLinkUnnamedModel8 = Field(default=None)
    status: IssueLinkUnnamedModel14 = Field(default=None)
    summary: str = Field(default=None)
    timetracking: IssueLinkUnnamedModel15 = Field(default=None)


class IssueLinkUnnamedModel16(IssueLinkFields):
    pass


class IssueLinkLinkedIssue(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fields: IssueLinkUnnamedModel16 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    self: str = Field(default=None)


class IssueLinkUnnamedModel17(IssueLinkLinkedIssue):
    pass


class IssueLinkUnnamedModel36(IssueLinkType):
    pass


class IssueLinkUnnamedModel34(IssueLinkFields1):
    pass


class IssueLinkLinkedIssue1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fields: IssueLinkUnnamedModel34 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    self: str = Field(default=None)


class IssueLinkUnnamedModel35(IssueLinkLinkedIssue1):
    pass


class IssueLink(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    inwardIssue: IssueLinkUnnamedModel17
    outwardIssue: IssueLinkUnnamedModel35
    self: str = Field(default=None)
    type_: IssueLinkUnnamedModel36 = Field(alias='type')


class IssueLinkTypes(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueLinkTypes: list[IssueLinkType] = Field(default=None)


class DateRangeFilterRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    dateAfter: str
    dateBefore: str


class ArchivedIssuesFilterRequest(_HtmlReprMixin, BaseModel):
    archivedBy: list[str] = Field(default=None)
    archivedDateRange: DateRangeFilterRequest = Field(default=None)
    issueTypes: list[str] = Field(default=None)
    projects: list[str] = Field(default=None)
    reporters: list[str] = Field(default=None)


class ExportArchivedIssuesTaskProgressResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fileUrl: str = Field(default=None)
    payload: str = Field(default=None)
    progress: int = Field(default=None)
    status: str = Field(default=None)
    submittedTime: datetime = Field(default=None)
    taskId: str = Field(default=None)


class SecurityLevel(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    isDefault: bool = Field(default=None)
    issueSecuritySchemeId: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class SecurityScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultSecurityLevelId: int = Field(default=None)
    description: str = Field(default=None)
    id: int = Field(default=None)
    levels: list[SecurityLevel] = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class SecuritySchemes(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueSecuritySchemes: list[SecurityScheme] = Field(default=None)


class SecuritySchemeLevelMemberBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    parameter: str = Field(default=None)
    type_: str = Field(alias='type')


class SecuritySchemeLevelBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None, max_length=4000)
    isDefault: bool = Field(default=None)
    members: list[SecuritySchemeLevelMemberBean] = Field(default=None)
    name: str = Field(max_length=255)


class CreateIssueSecuritySchemeDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None, max_length=255)
    levels: list[SecuritySchemeLevelBean] = Field(default=None)
    name: str = Field(max_length=60)


class SecuritySchemeId(_HtmlReprMixin, BaseModel):
    id: str


class PageBeanSecurityLevel(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[SecurityLevel] = Field(default=None)


class DefaultLevelValue(_HtmlReprMixin, BaseModel):
    defaultLevelId: str
    issueSecuritySchemeId: str


class SetDefaultLevelsRequest(_HtmlReprMixin, BaseModel):
    defaultValues: list[DefaultLevelValue] = Field(max_length=1000)


class PermissionHolder(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: str = Field(default=None)
    parameter: str = Field(default=None)
    type_: str = Field(alias='type')
    value: str = Field(default=None)


class SecurityLevelMemberUnnamedModel(PermissionHolder):
    pass


class SecurityLevelMember(_HtmlReprMixin, BaseModel):
    holder: SecurityLevelMemberUnnamedModel
    id: str
    issueSecurityLevelId: str
    issueSecuritySchemeId: str
    managed: bool = Field(default=None)


class PageBeanSecurityLevelMember(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[SecurityLevelMember] = Field(default=None)


class IssueSecuritySchemeToProjectMapping(_HtmlReprMixin, BaseModel):
    issueSecuritySchemeId: str = Field(default=None)
    projectId: str = Field(default=None)


class PageBeanIssueSecuritySchemeToProjectMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueSecuritySchemeToProjectMapping] = Field(default=None)


class OldToNewSecurityLevelMappingsBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    newLevelId: str
    oldLevelId: str


class AssociateSecuritySchemeWithProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    oldToNewSecurityLevelMappings: list[OldToNewSecurityLevelMappingsBean] = Field(
        default=None
    )
    projectId: str
    schemeId: str


class SecuritySchemeWithProjects(_HtmlReprMixin, BaseModel):
    defaultLevel: int = Field(default=None)
    description: str = Field(default=None)
    id: int
    name: str
    projectIds: list[int] = Field(default=None)
    self: str


class PageBeanSecuritySchemeWithProjects(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[SecuritySchemeWithProjects] = Field(default=None)


class UpdateIssueSecuritySchemeRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None, max_length=255)
    name: str = Field(default=None, max_length=60)


class IssueSecurityLevelMemberUnnamedModel(PermissionHolder):
    pass


class IssueSecurityLevelMember(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    holder: IssueSecurityLevelMemberUnnamedModel
    id: int
    issueSecurityLevelId: int


class PageBeanIssueSecurityLevelMember(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueSecurityLevelMember] = Field(default=None)


class AddSecuritySchemeLevelsRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    levels: list[SecuritySchemeLevelBean] = Field(default=None)


class UpdateIssueSecurityLevelDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None, max_length=255)
    name: str = Field(default=None, max_length=60)


class SecuritySchemeMembersRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    members: list[SecuritySchemeLevelMemberBean] = Field(default=None)


class IssueTypeCreateBeanType(_HtmlReprMixin, str, Enum):
    SUBTASK = 'subtask'
    STANDARD = 'standard'


class IssueTypeCreateBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    hierarchyLevel: int = Field(default=None)
    name: str
    type_: IssueTypeCreateBeanType = Field(default=None, alias='type')


class IssueTypeUpdateBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarId: int = Field(default=None)
    description: str = Field(default=None)
    name: str = Field(default=None)


class AutoEnum17(_HtmlReprMixin, str, Enum):
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'
    ID = 'id'
    ID_1 = '-id'
    ID_2 = '+id'


class IssueTypeScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultIssueTypeId: str = Field(default=None)
    description: str = Field(default=None)
    id: str
    isDefault: bool = Field(default=None)
    name: str


class PageBeanIssueTypeScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueTypeScheme] = Field(default=None)


class IssueTypeSchemeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultIssueTypeId: str = Field(default=None)
    description: str = Field(default=None)
    issueTypeIds: list[str]
    name: str


class IssueTypeSchemeID(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeSchemeId: str


class IssueTypeSchemeMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str
    issueTypeSchemeId: str


class PageBeanIssueTypeSchemeMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueTypeSchemeMapping] = Field(default=None)


class IssueTypeSchemeProjectsUnnamedModel(IssueTypeScheme):
    pass


class IssueTypeSchemeProjects(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeScheme: IssueTypeSchemeProjectsUnnamedModel
    projectIds: list[str]


class PageBeanIssueTypeSchemeProjects(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueTypeSchemeProjects] = Field(default=None)


class IssueTypeSchemeProjectAssociation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeSchemeId: str
    projectId: str


class IssueTypeSchemeUpdateDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultIssueTypeId: str = Field(default=None)
    description: str = Field(default=None)
    name: str = Field(default=None)


class OrderOfIssueTypes(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    after: str = Field(default=None)
    issueTypeIds: list[str]
    position: OrderOfCustomFieldOptionsPosition = Field(default=None)


class IssueTypeScreenScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str
    name: str


class PageBeanIssueTypeScreenScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueTypeScreenScheme] = Field(default=None)


class IssueTypeScreenSchemeMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str
    screenSchemeId: str


class IssueTypeScreenSchemeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    issueTypeMappings: list[IssueTypeScreenSchemeMapping]
    name: str


class IssueTypeScreenSchemeId(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str


class IssueTypeScreenSchemeItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str
    issueTypeScreenSchemeId: str
    screenSchemeId: str


class PageBeanIssueTypeScreenSchemeItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueTypeScreenSchemeItem] = Field(default=None)


class IssueTypeScreenSchemesProjectsUnnamedModel(IssueTypeScreenScheme):
    pass


class IssueTypeScreenSchemesProjects(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeScreenScheme: IssueTypeScreenSchemesProjectsUnnamedModel
    projectIds: list[str]


class PageBeanIssueTypeScreenSchemesProjects(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[IssueTypeScreenSchemesProjects] = Field(default=None)


class IssueTypeScreenSchemeProjectAssociation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeScreenSchemeId: str = Field(default=None)
    projectId: str = Field(default=None)


class IssueTypeScreenSchemeUpdateDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)


class IssueTypeScreenSchemeMappingDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeMappings: list[IssueTypeScreenSchemeMapping]


class UpdateDefaultScreenScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    screenSchemeId: str


class PageBeanProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ProjectDetails] = Field(default=None)


class FunctionReferenceData(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    displayName: str = Field(default=None)
    isList: AutoEnum13 = Field(default=None)
    supportsListAndSingleValueOperators: AutoEnum13 = Field(default=None)
    types: list[str] = Field(default=None)
    value: str = Field(default=None)


class FieldReferenceData(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    auto: AutoEnum13 = Field(default=None)
    cfid: str = Field(default=None)
    deprecated: AutoEnum13 = Field(default=None)
    deprecatedSearcherKey: str = Field(default=None)
    displayName: str = Field(default=None)
    operators: list[str] = Field(default=None)
    orderable: AutoEnum13 = Field(default=None)
    searchable: AutoEnum13 = Field(default=None)
    types: list[str] = Field(default=None)
    value: str = Field(default=None)


class JQLReferenceData(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    jqlReservedWords: list[str] = Field(default=None)
    visibleFieldNames: list[FieldReferenceData] = Field(default=None)
    visibleFunctionNames: list[FunctionReferenceData] = Field(default=None)


class SearchAutoCompleteFilter(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    includeCollapsedFields: bool = Field(default=False)
    projectIds: list[int] = Field(default=None)


class AutoCompleteSuggestion(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    displayName: str = Field(default=None)
    value: str = Field(default=None)


class AutoCompleteSuggestions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    results: list[AutoCompleteSuggestion] = Field(default=None)


class JqlFunctionPrecomputationBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    arguments: list[str] = Field(default=None)
    created: datetime = Field(default=None)
    error: str = Field(default=None)
    field: str = Field(default=None)
    functionKey: str = Field(default=None)
    functionName: str = Field(default=None)
    id: str = Field(default=None)
    operator: str = Field(default=None)
    updated: datetime = Field(default=None)
    used: datetime = Field(default=None)
    value: str = Field(default=None)


class PageBean2JqlFunctionPrecomputationBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[JqlFunctionPrecomputationBean] = Field(default=None)


class JqlFunctionPrecomputationUpdateBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    error: str = Field(default=None)
    id: str
    value: str = Field(default=None)


class JqlFunctionPrecomputationUpdateRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    values: list[JqlFunctionPrecomputationUpdateBean] = Field(default=None)


class JqlFunctionPrecomputationUpdateResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    notFoundPrecomputationIDs: list[str] = Field(default=None)


class JqlFunctionPrecomputationUpdateErrorResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errorMessages: list[str] = Field(default=None)
    notFoundPrecomputationIDs: list[str] = Field(default=None)


class JqlFunctionPrecomputationGetByIdRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    precomputationIDs: list[str] = Field(default=None)


class JqlFunctionPrecomputationGetByIdResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    notFoundPrecomputationIDs: list[str] = Field(default=None)
    precomputations: list[JqlFunctionPrecomputationBean] = Field(default=None)


class IssuesAndJQLQueries(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueIds: list[int]
    jqls: list[str]


class IssueMatchesForJQL(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errors: list[str]
    matchedIssues: list[int]


class IssueMatches(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    matches: list[IssueMatchesForJQL]


class JqlQueriesToParse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    queries: list[str] = Field(min_length=1)


class JqlQueryFieldEntityPropertyType(_HtmlReprMixin, str, Enum):
    NUMBER = 'number'
    STRING = 'string'
    TEXT = 'text'
    DATE = 'date'
    USER = 'user'


class JqlQueryFieldEntityProperty(_HtmlReprMixin, BaseModel):
    entity: str
    key: str
    path: str
    type_: JqlQueryFieldEntityPropertyType = Field(default=None, alias='type')


class JqlQueryField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    encodedName: str = Field(default=None)
    name: str
    property: list[JqlQueryFieldEntityProperty] = Field(default=None)


class JqlQueryOrderByClauseElementDirection(_HtmlReprMixin, str, Enum):
    ASC = 'asc'
    DESC = 'desc'


class JqlQueryOrderByClauseElement(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    direction: JqlQueryOrderByClauseElementDirection = Field(default=None)
    field: JqlQueryField


class JqlQueryOrderByClause(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fields: list[JqlQueryOrderByClauseElement]


class JqlQuery(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    orderBy: JqlQueryOrderByClause = Field(default=None)
    where: CompoundClause | FieldValueClause | FieldWasClause | FieldChangedClause = (
        Field(default=None)
    )


class ParsedJqlQueryUnnamedModel(JqlQuery):
    pass


class ParsedJqlQuery(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errors: list[str] = Field(default=None)
    query: str
    structure: ParsedJqlQueryUnnamedModel = Field(default=None)
    warnings: list[str] = Field(default=None)


class ParsedJqlQueries(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    queries: list[ParsedJqlQuery] = Field(min_length=1)


class CompoundClauseOperator(_HtmlReprMixin, str, Enum):
    AND = 'and'
    OR = 'or'
    NOT = 'not'


class CompoundClause(_HtmlReprMixin, BaseModel):
    clauses: list[
        CompoundClause | FieldValueClause | FieldWasClause | FieldChangedClause
    ]
    operator: CompoundClauseOperator


class FieldValueClauseOperator(_HtmlReprMixin, str, Enum):
    UNNAMEDTYPE = '='
    UNNAMEDTYPE_1 = '!='
    UNNAMEDTYPE_2 = '>'
    UNNAMEDTYPE_3 = '<'
    UNNAMEDTYPE_4 = '>='
    UNNAMEDTYPE_5 = '<='
    IN = 'in'
    NOTIN = 'not in'
    UNNAMEDTYPE_6 = '~'
    UNNAMEDTYPE_7 = '~='
    IS = 'is'
    ISNOT = 'is not'


class FieldValueClause(_HtmlReprMixin, BaseModel):
    field: JqlQueryField
    operand: ListOperand | ValueOperand | FunctionOperand | KeywordOperand
    operator: FieldValueClauseOperator


class ListOperand(_HtmlReprMixin, BaseModel):
    encodedOperand: str = Field(default=None)
    values: list[ValueOperand | FunctionOperand | KeywordOperand] = Field(min_length=1)


class ValueOperand(_HtmlReprMixin, BaseModel):
    encodedValue: str = Field(default=None)
    value: str


class FunctionOperand(_HtmlReprMixin, BaseModel):
    arguments: list[str]
    encodedOperand: str = Field(default=None)
    function: str


class KeywordOperandKeyword(_HtmlReprMixin, str, Enum):
    EMPTY = 'empty'


class KeywordOperand(_HtmlReprMixin, BaseModel):
    keyword: KeywordOperandKeyword


class FieldWasClauseOperator(_HtmlReprMixin, str, Enum):
    WAS = 'was'
    WASIN = 'was in'
    WASNOTIN = 'was not in'
    WASNOT = 'was not'


class JqlQueryClauseTimePredicateOperator(_HtmlReprMixin, str, Enum):
    BEFORE = 'before'
    AFTER = 'after'
    FROM = 'from'
    TO = 'to'
    ON = 'on'
    DURING = 'during'
    BY = 'by'


class JqlQueryClauseTimePredicate(_HtmlReprMixin, BaseModel):
    operand: ListOperand | ValueOperand | FunctionOperand | KeywordOperand
    operator: JqlQueryClauseTimePredicateOperator


class FieldWasClause(_HtmlReprMixin, BaseModel):
    field: JqlQueryField
    operand: ListOperand | ValueOperand | FunctionOperand | KeywordOperand
    operator: FieldWasClauseOperator
    predicates: list[JqlQueryClauseTimePredicate]


class FieldChangedClauseOperator(_HtmlReprMixin, str, Enum):
    CHANGED = 'changed'


class FieldChangedClause(_HtmlReprMixin, BaseModel):
    field: JqlQueryField
    operator: FieldChangedClauseOperator
    predicates: list[JqlQueryClauseTimePredicate]


class JQLPersonalDataMigrationRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    queryStrings: list[str] = Field(default=None)


class JQLQueryWithUnknownUsers(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    convertedQuery: str = Field(default=None)
    originalQuery: str = Field(default=None)


class ConvertedJQLQueries(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    queriesWithUnknownUsers: list[JQLQueryWithUnknownUsers] = Field(default=None)
    queryStrings: list[str] = Field(default=None)


class JqlQueryToSanitize(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str | None = Field(default=None, max_length=128)
    query: str


class JqlQueriesToSanitize(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    queries: list[JqlQueryToSanitize]


class SanitizedJqlQueryUnnamedModel(ErrorCollection):
    pass


class SanitizedJqlQuery(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str | None = Field(default=None, max_length=128)
    errors: SanitizedJqlQueryUnnamedModel = Field(default=None)
    initialQuery: str = Field(default=None)
    sanitizedQuery: str | None = Field(default=None)


class SanitizedJqlQueries(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    queries: list[SanitizedJqlQuery] = Field(default=None)


class PageBeanString(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[str] = Field(default=None)


class LicenseMetric(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    key: str = Field(default=None)
    value: str = Field(default=None)


class AutoEnum18(_HtmlReprMixin, str, Enum):
    JIRACORE = 'jira-core'
    JIRAPRODUCTDISCOVERY = 'jira-product-discovery'
    JIRASOFTWARE = 'jira-software'
    JIRASERVICEDESK = 'jira-servicedesk'


class UserPermissionType(_HtmlReprMixin, str, Enum):
    GLOBAL = 'GLOBAL'
    PROJECT = 'PROJECT'


class UserPermission(_HtmlReprMixin, BaseModel):
    deprecatedKey: bool = Field(default=None)
    description: str = Field(default=None)
    havePermission: bool = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    type_: UserPermissionType = Field(default=None, alias='type')


class Permissions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    permissions: dict[str, UserPermission] = Field(default=None)


class Locale(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    locale: str = Field(default=None)


class EventNotificationUnnamedModel8(UpdatedProjectCategory):
    pass


class EventNotificationUnnamedModel7(AvatarUrlsBean):
    pass


class EventNotificationProjectDetails1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: EventNotificationUnnamedModel7 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: EventNotificationUnnamedModel8 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class EventNotificationUnnamedModel9(EventNotificationProjectDetails1):
    pass


class EventNotificationUnnamedModel12(AvatarUrlsBean):
    pass


class EventNotificationUserDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: str = Field(default=None)
    active: bool = Field(default=None)
    avatarUrls: EventNotificationUnnamedModel12 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class EventNotificationUnnamedModel13(EventNotificationUserDetails):
    pass


class EventNotificationUnnamedModel1(AvatarUrlsBean):
    pass


class EventNotificationUnnamedModel2(UpdatedProjectCategory):
    pass


class EventNotificationProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: EventNotificationUnnamedModel1 = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: EventNotificationUnnamedModel2 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class EventNotificationUnnamedModel3(EventNotificationProjectDetails):
    pass


class EventNotificationScope(_HtmlReprMixin, BaseModel):
    project: EventNotificationUnnamedModel3 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class EventNotificationScope1(_HtmlReprMixin, BaseModel):
    project: EventNotificationUnnamedModel9 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class NotificationEvent(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)
    templateEvent: NotificationEventUnnamedModel = Field(default=None)


class NotificationEventUnnamedModel(NotificationEvent):
    pass


class EventNotificationUnnamedModel10(EventNotificationScope1):
    pass


class EventNotificationProjectRole(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    actors: list[RoleActor] = Field(default=None)
    admin: bool = Field(default=None)
    currentUserRole: bool = Field(default=None)
    default: bool = Field(default=None)
    description: str = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)
    roleConfigurable: bool = Field(default=None)
    scope: EventNotificationUnnamedModel10 = Field(default=None)
    self: str = Field(default=None)
    translatedName: str = Field(default=None)


class EventNotificationUnnamedModel11(EventNotificationProjectRole):
    pass


class EventNotificationUnnamedModel(JsonTypeBean):
    pass


class EventNotificationUnnamedModel4(EventNotificationScope):
    pass


class EventNotificationFieldDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    clauseNames: list[str] = Field(default=None)
    custom: bool = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    navigable: bool = Field(default=None)
    orderable: bool = Field(default=None)
    schema: EventNotificationUnnamedModel = Field(default=None)
    scope: EventNotificationUnnamedModel4 = Field(default=None)
    searchable: bool = Field(default=None)


class EventNotificationUnnamedModel5(EventNotificationFieldDetails):
    pass


class EventNotificationUnnamedModel6(GroupName):
    pass


class EventNotificationNotificationType(_HtmlReprMixin, str, Enum):
    CURRENTASSIGNEE = 'CurrentAssignee'
    REPORTER = 'Reporter'
    CURRENTUSER = 'CurrentUser'
    PROJECTLEAD = 'ProjectLead'
    COMPONENTLEAD = 'ComponentLead'
    USER = 'User'
    GROUP = 'Group'
    PROJECTROLE = 'ProjectRole'
    EMAILADDRESS = 'EmailAddress'
    ALLWATCHERS = 'AllWatchers'
    USERCUSTOMFIELD = 'UserCustomField'
    GROUPCUSTOMFIELD = 'GroupCustomField'


class EventNotification(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    field: EventNotificationUnnamedModel5 = Field(default=None)
    group: EventNotificationUnnamedModel6 = Field(default=None)
    id: int = Field(default=None)
    notificationType: EventNotificationNotificationType = Field(default=None)
    parameter: str = Field(default=None)
    projectRole: EventNotificationUnnamedModel11 = Field(default=None)
    recipient: str = Field(default=None)
    user: EventNotificationUnnamedModel13 = Field(default=None)


class NotificationSchemeUnnamedModel(AvatarUrlsBean):
    pass


class NotificationSchemeUnnamedModel1(UpdatedProjectCategory):
    pass


class NotificationSchemeProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: NotificationSchemeUnnamedModel = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: NotificationSchemeUnnamedModel1 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class NotificationSchemeUnnamedModel2(NotificationSchemeProjectDetails):
    pass


class NotificationSchemeScope(_HtmlReprMixin, BaseModel):
    project: NotificationSchemeUnnamedModel2 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class NotificationSchemeEvent(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    event: NotificationEvent = Field(default=None)
    notifications: list[EventNotification] = Field(default=None)


class NotificationSchemeUnnamedModel3(NotificationSchemeScope):
    pass


class NotificationScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    expand: str = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)
    notificationSchemeEvents: list[NotificationSchemeEvent] = Field(default=None)
    projects: list[int] = Field(default=None)
    scope: NotificationSchemeUnnamedModel3 = Field(default=None)
    self: str = Field(default=None)


class PageBeanNotificationScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[NotificationScheme] = Field(default=None)


class NotificationSchemeEventTypeId(_HtmlReprMixin, BaseModel):
    id: str


class NotificationSchemeNotificationDetails(_HtmlReprMixin, BaseModel):
    notificationType: str
    parameter: str = Field(default=None)


class NotificationSchemeEventDetailsUnnamedModel(NotificationSchemeEventTypeId):
    pass


class NotificationSchemeEventDetails(_HtmlReprMixin, BaseModel):
    event: NotificationSchemeEventDetailsUnnamedModel
    notifications: list[NotificationSchemeNotificationDetails] = Field(max_length=255)


class CreateNotificationSchemeDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None, max_length=4000)
    name: str = Field(max_length=255)
    notificationSchemeEvents: list[NotificationSchemeEventDetails] = Field(default=None)


class NotificationSchemeId(_HtmlReprMixin, BaseModel):
    id: str


class NotificationSchemeAndProjectMappingJsonBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    notificationSchemeId: str = Field(default=None)
    projectId: str = Field(default=None)


class PageBeanNotificationSchemeAndProjectMappingJsonBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[NotificationSchemeAndProjectMappingJsonBean] = Field(default=None)


class UpdateNotificationSchemeDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None, max_length=4000)
    name: str = Field(default=None, max_length=255)


class AddNotificationsDetails(_HtmlReprMixin, BaseModel):
    notificationSchemeEvents: list[NotificationSchemeEventDetails]


class BulkProjectPermissions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issues: list[int] = Field(default=None)
    permissions: list[str]
    projects: list[int] = Field(default=None)


class BulkPermissionsRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None)
    globalPermissions: list[str] = Field(default=None)
    projectPermissions: list[BulkProjectPermissions] = Field(default=None)


class BulkProjectPermissionGrants(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issues: list[int]
    permission: str
    projects: list[int]


class BulkPermissionGrants(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    globalPermissions: list[str]
    projectPermissions: list[BulkProjectPermissionGrants]


class PermissionsKeysBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    permissions: list[str]


class ProjectIdentifierBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(default=None)
    key: str = Field(default=None)


class PermittedProjects(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projects: list[ProjectIdentifierBean] = Field(default=None)


class PermissionGrantUnnamedModel(PermissionHolder):
    pass


class PermissionGrant(_HtmlReprMixin, BaseModel):
    holder: PermissionGrantUnnamedModel = Field(default=None)
    id: int = Field(default=None)
    permission: str = Field(default=None)
    self: str = Field(default=None)


class PermissionSchemeUnnamedModel(AvatarUrlsBean):
    pass


class PermissionSchemeUnnamedModel1(UpdatedProjectCategory):
    pass


class PermissionSchemeProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: PermissionSchemeUnnamedModel = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: PermissionSchemeUnnamedModel1 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class PermissionSchemeUnnamedModel2(PermissionSchemeProjectDetails):
    pass


class PermissionSchemeScope(_HtmlReprMixin, BaseModel):
    project: PermissionSchemeUnnamedModel2 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class PermissionSchemeUnnamedModel3(PermissionSchemeScope):
    pass


class PermissionScheme(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None)
    expand: str = Field(default=None)
    id: int = Field(default=None)
    name: str
    permissions: list[PermissionGrant] = Field(default=None)
    scope: PermissionSchemeUnnamedModel3 = Field(default=None)
    self: str = Field(default=None)


class PermissionSchemes(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    permissionSchemes: list[PermissionScheme] = Field(default=None)


class PermissionGrants(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: str = Field(default=None)
    permissions: list[PermissionGrant] = Field(default=None)


class GetPlanResponseForPageStatus(_HtmlReprMixin, str, Enum):
    ACTIVE = 'Active'
    TRASHED = 'Trashed'
    ARCHIVED = 'Archived'


class GetIssueSourceResponseType(_HtmlReprMixin, str, Enum):
    BOARD = 'Board'
    PROJECT = 'Project'
    FILTER = 'Filter'
    CUSTOM = 'Custom'


class GetIssueSourceResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    type_: GetIssueSourceResponseType = Field(alias='type')
    value: int


class GetPlanResponseForPage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str
    issueSources: list[GetIssueSourceResponse] = Field(default=None)
    name: str
    scenarioId: str
    status: GetPlanResponseForPageStatus


class PageWithCursorGetPlanResponseForPage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    cursor: str = Field(default=None)
    last: bool = Field(default=None)
    nextPageCursor: str = Field(default=None)
    size: int = Field(default=None)
    total: int = Field(default=None)
    values: list[GetPlanResponseForPage] = Field(default=None)


class CreatePermissionRequestType(_HtmlReprMixin, str, Enum):
    GROUP = 'Group'
    ACCOUNTID = 'AccountId'


class CreatePermissionHolderRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    type_: CreatePermissionRequestType = Field(alias='type')
    value: str


class CreatePlanRequestType(_HtmlReprMixin, str, Enum):
    DUEDATE = 'DueDate'
    TARGETSTARTDATE = 'TargetStartDate'
    TARGETENDDATE = 'TargetEndDate'
    DATECUSTOMFIELD = 'DateCustomField'


class CreateExclusionRulesRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueIds: list[int] = Field(default=None)
    issueTypeIds: list[int] = Field(default=None)
    numberOfDaysToShowCompletedIssues: int = Field(default=None)
    releaseIds: list[int] = Field(default=None)
    workStatusCategoryIds: list[int] = Field(default=None)
    workStatusIds: list[int] = Field(default=None)


class CreatePlanRequestUnnamedModel(CreateExclusionRulesRequest):
    pass


class CreatePermissionRequestUnnamedModel(CreatePermissionHolderRequest):
    pass


class CreatePermissionRequestType1(_HtmlReprMixin, str, Enum):
    VIEW = 'View'
    EDIT = 'Edit'


class CreatePermissionRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    holder: CreatePermissionRequestUnnamedModel
    type_: CreatePermissionRequestType1 = Field(alias='type')


class CreateDateFieldRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    dateCustomFieldId: int = Field(default=None)
    type_: CreatePlanRequestType = Field(alias='type')


class CreatePlanRequestUnnamedModel2(CreateDateFieldRequest):
    pass


class CreateCustomFieldRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customFieldId: int
    filter: bool = Field(default=None)


class CreateIssueSourceRequestType(_HtmlReprMixin, str, Enum):
    BOARD = 'Board'
    PROJECT = 'Project'
    FILTER = 'Filter'


class CreatePlanRequestUnnamedModel1(CreateDateFieldRequest):
    pass


class CreatePlanRequestInferredDates(_HtmlReprMixin, str, Enum):
    NONE = 'None'
    SPRINTDATES = 'SprintDates'
    RELEASEDATES = 'ReleaseDates'


class CreateIssueSourceRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    type_: CreateIssueSourceRequestType = Field(alias='type')
    value: int


class CreatePlanRequestDependencies(_HtmlReprMixin, str, Enum):
    SEQUENTIAL = 'Sequential'
    CONCURRENT = 'Concurrent'


class CreatePlanRequestEstimation(_HtmlReprMixin, str, Enum):
    STORYPOINTS = 'StoryPoints'
    DAYS = 'Days'
    HOURS = 'Hours'


class CreateSchedulingRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    dependencies: CreatePlanRequestDependencies = Field(default=None)
    endDate: CreatePlanRequestUnnamedModel1 = Field(default=None)
    estimation: CreatePlanRequestEstimation
    inferredDates: CreatePlanRequestInferredDates = Field(default=None)
    startDate: CreatePlanRequestUnnamedModel2 = Field(default=None)


class CreateCrossProjectReleaseRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    name: str
    releaseIds: list[int] = Field(default=None)


class CreatePlanRequestUnnamedModel3(CreateSchedulingRequest):
    pass


class CreatePlanRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    crossProjectReleases: list[CreateCrossProjectReleaseRequest] = Field(default=None)
    customFields: list[CreateCustomFieldRequest] = Field(default=None)
    exclusionRules: CreatePlanRequestUnnamedModel = Field(default=None)
    issueSources: list[CreateIssueSourceRequest]
    leadAccountId: str = Field(default=None)
    name: str = Field(min_length=1, max_length=255)
    permissions: list[CreatePermissionRequest] = Field(default=None)
    scheduling: CreatePlanRequestUnnamedModel3


class GetCrossProjectReleaseResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    name: str = Field(default=None)
    releaseIds: list[int] = Field(default=None)


class GetDateFieldResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    dateCustomFieldId: int = Field(default=None)
    type_: CreatePlanRequestType = Field(alias='type')


class GetPlanResponseUnnamedModel2(GetDateFieldResponse):
    pass


class GetPlanResponseUnnamedModel1(GetDateFieldResponse):
    pass


class GetSchedulingResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    dependencies: CreatePlanRequestDependencies
    endDate: GetPlanResponseUnnamedModel1
    estimation: CreatePlanRequestEstimation
    inferredDates: CreatePlanRequestInferredDates
    startDate: GetPlanResponseUnnamedModel2


class GetExclusionRulesResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueIds: list[int] = Field(default=None)
    issueTypeIds: list[int] = Field(default=None)
    numberOfDaysToShowCompletedIssues: int
    releaseIds: list[int] = Field(default=None)
    workStatusCategoryIds: list[int] = Field(default=None)
    workStatusIds: list[int] = Field(default=None)


class GetPermissionHolderResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    type_: CreatePermissionRequestType = Field(alias='type')
    value: str


class GetPermissionResponseUnnamedModel(GetPermissionHolderResponse):
    pass


class GetPermissionResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    holder: GetPermissionResponseUnnamedModel
    type_: CreatePermissionRequestType1 = Field(alias='type')


class GetPlanResponseUnnamedModel(GetExclusionRulesResponse):
    pass


class GetCustomFieldResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customFieldId: int
    filter: bool = Field(default=None)


class GetPlanResponseUnnamedModel3(GetSchedulingResponse):
    pass


class GetPlanResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    crossProjectReleases: list[GetCrossProjectReleaseResponse] = Field(default=None)
    customFields: list[GetCustomFieldResponse] = Field(default=None)
    exclusionRules: GetPlanResponseUnnamedModel = Field(default=None)
    id: int
    issueSources: list[GetIssueSourceResponse] = Field(default=None)
    lastSaved: str = Field(default=None)
    leadAccountId: str = Field(default=None)
    name: str = Field(default=None)
    permissions: list[GetPermissionResponse] = Field(default=None)
    scheduling: GetPlanResponseUnnamedModel3
    status: GetPlanResponseForPageStatus


class DuplicatePlanRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    name: str


class GetTeamResponseForPageType(_HtmlReprMixin, str, Enum):
    PLANONLY = 'PlanOnly'
    ATLASSIAN = 'Atlassian'


class GetTeamResponseForPage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str
    name: str = Field(default=None)
    type_: GetTeamResponseForPageType = Field(alias='type')


class PageWithCursorGetTeamResponseForPage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    cursor: str = Field(default=None)
    last: bool = Field(default=None)
    nextPageCursor: str = Field(default=None)
    size: int = Field(default=None)
    total: int = Field(default=None)
    values: list[GetTeamResponseForPage] = Field(default=None)


class AddAtlassianTeamRequestPlanningStyle(_HtmlReprMixin, str, Enum):
    SCRUM = 'Scrum'
    KANBAN = 'Kanban'


class AddAtlassianTeamRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    capacity: float = Field(default=None)
    id: str
    issueSourceId: int = Field(default=None)
    planningStyle: AddAtlassianTeamRequestPlanningStyle
    sprintLength: int = Field(default=None)


class GetAtlassianTeamResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    capacity: float = Field(default=None)
    id: str
    issueSourceId: int = Field(default=None)
    planningStyle: AddAtlassianTeamRequestPlanningStyle
    sprintLength: int = Field(default=None)


class CreatePlanOnlyTeamRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    capacity: float = Field(default=None)
    issueSourceId: int = Field(default=None)
    memberAccountIds: list[str] = Field(default=None)
    name: str = Field(min_length=1, max_length=255)
    planningStyle: AddAtlassianTeamRequestPlanningStyle
    sprintLength: int = Field(default=None)


class GetPlanOnlyTeamResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    capacity: float = Field(default=None)
    id: int
    issueSourceId: int = Field(default=None)
    memberAccountIds: list[str] = Field(default=None)
    name: str
    planningStyle: AddAtlassianTeamRequestPlanningStyle
    sprintLength: int = Field(default=None)


class CreatePriorityDetailsIconUrl(_HtmlReprMixin, str, Enum):
    IMAGESICONSPRIORITIESBLOCKERPNG = '/images/icons/priorities/blocker.png'
    IMAGESICONSPRIORITIESCRITICALPNG = '/images/icons/priorities/critical.png'
    IMAGESICONSPRIORITIESHIGHPNG = '/images/icons/priorities/high.png'
    IMAGESICONSPRIORITIESHIGHESTPNG = '/images/icons/priorities/highest.png'
    IMAGESICONSPRIORITIESLOWPNG = '/images/icons/priorities/low.png'
    IMAGESICONSPRIORITIESLOWESTPNG = '/images/icons/priorities/lowest.png'
    IMAGESICONSPRIORITIESMAJORPNG = '/images/icons/priorities/major.png'
    IMAGESICONSPRIORITIESMEDIUMPNG = '/images/icons/priorities/medium.png'
    IMAGESICONSPRIORITIESMINORPNG = '/images/icons/priorities/minor.png'
    IMAGESICONSPRIORITIESTRIVIALPNG = '/images/icons/priorities/trivial.png'
    IMAGESICONSPRIORITIESBLOCKERNEWPNG = '/images/icons/priorities/blocker_new.png'
    IMAGESICONSPRIORITIESCRITICALNEWPNG = '/images/icons/priorities/critical_new.png'
    IMAGESICONSPRIORITIESHIGHNEWPNG = '/images/icons/priorities/high_new.png'
    IMAGESICONSPRIORITIESHIGHESTNEWPNG = '/images/icons/priorities/highest_new.png'
    IMAGESICONSPRIORITIESLOWNEWPNG = '/images/icons/priorities/low_new.png'
    IMAGESICONSPRIORITIESLOWESTNEWPNG = '/images/icons/priorities/lowest_new.png'
    IMAGESICONSPRIORITIESMAJORNEWPNG = '/images/icons/priorities/major_new.png'
    IMAGESICONSPRIORITIESMEDIUMNEWPNG = '/images/icons/priorities/medium_new.png'
    IMAGESICONSPRIORITIESMINORNEWPNG = '/images/icons/priorities/minor_new.png'
    IMAGESICONSPRIORITIESTRIVIALNEWPNG = '/images/icons/priorities/trivial_new.png'


class CreatePriorityDetails(_HtmlReprMixin, BaseModel):
    avatarId: int = Field(default=None)
    description: str | None = Field(default=None, max_length=255)
    iconUrl: CreatePriorityDetailsIconUrl | None = Field(default=None, max_length=255)
    name: str = Field(max_length=60)
    statusColor: str


class PriorityId(_HtmlReprMixin, BaseModel):
    id: str


class SetDefaultPriorityRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str


class ReorderIssuePriorities(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    after: str = Field(default=None)
    ids: list[str]
    position: str = Field(default=None)


class PageBeanPriority(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Priority] = Field(default=None)


class UpdatePriorityDetails(_HtmlReprMixin, BaseModel):
    avatarId: int = Field(default=None)
    description: str | None = Field(default=None, max_length=255)
    iconUrl: CreatePriorityDetailsIconUrl | None = Field(default=None, max_length=255)
    name: str | None = Field(default=None, max_length=60)
    statusColor: str | None = Field(default=None)


class AutoEnum19(_HtmlReprMixin, str, Enum):
    NAME = 'name'
    NAME_1 = '+name'
    NAME_2 = '-name'


class PriorityWithSequence(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    isDefault: bool = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    sequence: str = Field(default=None)
    statusColor: str = Field(default=None)


class PageBeanPriorityWithSequence(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[PriorityWithSequence] = Field(default=None)


class PrioritySchemeWithPaginatedPrioritiesAndProjectsUnnamedModel(
    PageBeanPriorityWithSequence
):
    pass


class PrioritySchemeWithPaginatedPrioritiesAndProjectsUnnamedModel1(
    PageBeanProjectDetails
):
    pass


class PrioritySchemeWithPaginatedPrioritiesAndProjects(_HtmlReprMixin, BaseModel):
    default: bool = Field(default=None)
    defaultPriorityId: str = Field(default=None)
    description: str = Field(default=None)
    id: str
    isDefault: bool = Field(default=None)
    name: str
    priorities: PrioritySchemeWithPaginatedPrioritiesAndProjectsUnnamedModel = Field(
        default=None
    )
    projects: PrioritySchemeWithPaginatedPrioritiesAndProjectsUnnamedModel1 = Field(
        default=None
    )
    self: str = Field(default=None)


class PageBeanPrioritySchemeWithPaginatedPrioritiesAndProjects(
    _HtmlReprMixin, BaseModel
):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[PrioritySchemeWithPaginatedPrioritiesAndProjects] = Field(default=None)


class PriorityMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    in_: dict[str, int] = Field(default=None, alias='in')
    out: dict[str, int] = Field(default=None)


class CreatePrioritySchemeDetailsUnnamedModel(PriorityMapping):
    pass


class CreatePrioritySchemeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultPriorityId: int
    description: str = Field(default=None, max_length=4000)
    mappings: CreatePrioritySchemeDetailsUnnamedModel = Field(default=None)
    name: str = Field(max_length=255)
    priorityIds: list[int] = Field(min_length=1, max_length=300)
    projectIds: list[int] = Field(default=None)


class PrioritySchemeIdUnnamedModel(JsonNode):
    pass


class TaskProgressBeanJsonNode(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    elapsedRuntime: int
    finished: int = Field(default=None)
    id: str
    lastUpdate: int
    message: str = Field(default=None)
    progress: int
    result: PrioritySchemeIdUnnamedModel = Field(default=None)
    self: str
    started: int = Field(default=None)
    status: BulkOperationProgressStatus
    submitted: int
    submittedBy: int


class PrioritySchemeIdUnnamedModel1(TaskProgressBeanJsonNode):
    pass


class PrioritySchemeId(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    task: PrioritySchemeIdUnnamedModel1 = Field(default=None)


class SuggestedMappingsForProjectsRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    add: list[int] = Field(default=None)


class SuggestedMappingsRequestBeanUnnamedModel1(
    SuggestedMappingsForProjectsRequestBean
):
    pass


class SuggestedMappingsForPrioritiesRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    add: list[int] = Field(default=None)
    remove: list[int] = Field(default=None)


class SuggestedMappingsRequestBeanUnnamedModel(
    SuggestedMappingsForPrioritiesRequestBean
):
    pass


class SuggestedMappingsRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    maxResults: int = Field(default=None)
    priorities: SuggestedMappingsRequestBeanUnnamedModel = Field(default=None)
    projects: SuggestedMappingsRequestBeanUnnamedModel1 = Field(default=None)
    schemeId: int = Field(default=None)
    startAt: int = Field(default=None)


class UpdatePrioritySchemeRequestBeanUnnamedModel(PriorityMapping):
    pass


class PrioritySchemeChangesWithoutMappings(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    ids: list[int]


class UpdatePrioritySchemeRequestBeanUnnamedModel4(
    PrioritySchemeChangesWithoutMappings
):
    pass


class UpdatePrioritySchemeRequestBeanUnnamedModel2(
    PrioritySchemeChangesWithoutMappings
):
    pass


class UpdatePrioritySchemeRequestBeanUnnamedModel1(
    PrioritySchemeChangesWithoutMappings
):
    pass


class UpdatePrioritiesInSchemeRequestBean(_HtmlReprMixin, BaseModel):
    add: UpdatePrioritySchemeRequestBeanUnnamedModel1 = Field(default=None)
    remove: UpdatePrioritySchemeRequestBeanUnnamedModel2 = Field(default=None)


class UpdatePrioritySchemeRequestBeanUnnamedModel3(UpdatePrioritiesInSchemeRequestBean):
    pass


class UpdatePrioritySchemeRequestBeanUnnamedModel5(
    PrioritySchemeChangesWithoutMappings
):
    pass


class UpdateProjectsInSchemeRequestBean(_HtmlReprMixin, BaseModel):
    add: UpdatePrioritySchemeRequestBeanUnnamedModel4 = Field(default=None)
    remove: UpdatePrioritySchemeRequestBeanUnnamedModel5 = Field(default=None)


class UpdatePrioritySchemeRequestBeanUnnamedModel6(UpdateProjectsInSchemeRequestBean):
    pass


class UpdatePrioritySchemeRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultPriorityId: int = Field(default=None)
    description: str = Field(default=None, max_length=4000)
    mappings: UpdatePrioritySchemeRequestBeanUnnamedModel = Field(default=None)
    name: str = Field(default=None, max_length=255)
    priorities: UpdatePrioritySchemeRequestBeanUnnamedModel3 = Field(default=None)
    projects: UpdatePrioritySchemeRequestBeanUnnamedModel6 = Field(default=None)


class UpdatePrioritySchemeResponseBeanUnnamedModel(JsonNode):
    pass


class UpdatePrioritySchemeResponseBeanTaskProgressBeanJsonNode(
    _HtmlReprMixin, BaseModel
):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    elapsedRuntime: int
    finished: int = Field(default=None)
    id: str
    lastUpdate: int
    message: str = Field(default=None)
    progress: int
    result: UpdatePrioritySchemeResponseBeanUnnamedModel = Field(default=None)
    self: str
    started: int = Field(default=None)
    status: BulkOperationProgressStatus
    submitted: int
    submittedBy: int


class UpdatePrioritySchemeResponseBeanUnnamedModel1(
    UpdatePrioritySchemeResponseBeanTaskProgressBeanJsonNode
):
    pass


class UpdatePrioritySchemeResponseBean(_HtmlReprMixin, BaseModel):
    priorityScheme: PrioritySchemeWithPaginatedPrioritiesAndProjects = Field(
        default=None
    )
    task: UpdatePrioritySchemeResponseBeanUnnamedModel1 = Field(default=None)


class PageBeanProject(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Project] = Field(default=None)


class CreateProjectDetailsProjectTemplateKey(_HtmlReprMixin, str, Enum):
    COMPYXISGREENHOPPERJIRAGHSIMPLIFIEDAGILITYKANBAN = (
        'com.pyxis.greenhopper.jira:gh-simplified-agility-kanban'
    )
    COMPYXISGREENHOPPERJIRAGHSIMPLIFIEDAGILITYSCRUM = (
        'com.pyxis.greenhopper.jira:gh-simplified-agility-scrum'
    )
    COMPYXISGREENHOPPERJIRAGHSIMPLIFIEDBASIC = (
        'com.pyxis.greenhopper.jira:gh-simplified-basic'
    )
    COMPYXISGREENHOPPERJIRAGHSIMPLIFIEDKANBANCLASSIC = (
        'com.pyxis.greenhopper.jira:gh-simplified-kanban-classic'
    )
    COMPYXISGREENHOPPERJIRAGHSIMPLIFIEDSCRUMCLASSIC = (
        'com.pyxis.greenhopper.jira:gh-simplified-scrum-classic'
    )
    COMPYXISGREENHOPPERJIRAGHCROSSTEAMTEMPLATE = (
        'com.pyxis.greenhopper.jira:gh-cross-team-template'
    )
    COMPYXISGREENHOPPERJIRAGHCROSSTEAMPLANNINGTEMPLATE = (
        'com.pyxis.greenhopper.jira:gh-cross-team-planning-template'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDITSERVICEMANAGEMENT = (
        'com.atlassian.servicedesk:simplified-it-service-management'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDITSERVICEMANAGEMENTBASIC = (
        'com.atlassian.servicedesk:simplified-it-service-management-basic'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDITSERVICEMANAGEMENTOPERATIONS = (
        'com.atlassian.servicedesk:simplified-it-service-management-operations'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDINTERNALSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-internal-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDEXTERNALSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-external-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDHRSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-hr-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDFACILITIESSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-facilities-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDLEGALSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-legal-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDMARKETINGSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-marketing-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDFINANCESERVICEDESK = (
        'com.atlassian.servicedesk:simplified-finance-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDANALYTICSSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-analytics-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDDESIGNSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-design-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDSALESSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-sales-service-desk'
    )
    COMATLASSIANSERVICEDESKSIMPLIFIEDHALPSERVICEDESK = (
        'com.atlassian.servicedesk:simplified-halp-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENITSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-it-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENHRSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-hr-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENLEGALSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-legal-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENMARKETINGSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-marketing-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENFACILITIESSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-facilities-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENGENERALSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-general-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENANALYTICSSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-analytics-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENFINANCESERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-finance-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENDESIGNSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-design-service-desk'
    )
    COMATLASSIANSERVICEDESKNEXTGENSALESSERVICEDESK = (
        'com.atlassian.servicedesk:next-gen-sales-service-desk'
    )
    COMATLASSIANJIRACOREPROJECTTEMPLATESJIRACORESIMPLIFIEDCONTENTMANAGEMENT = 'com.atlassian.jira-core-project-templates:jira-core-simplified-content-management'
    COMATLASSIANJIRACOREPROJECTTEMPLATESJIRACORESIMPLIFIEDDOCUMENTAPPROVAL = 'com.atlassian.jira-core-project-templates:jira-core-simplified-document-approval'
    COMATLASSIANJIRACOREPROJECTTEMPLATESJIRACORESIMPLIFIEDLEADTRACKING = (
        'com.atlassian.jira-core-project-templates:jira-core-simplified-lead-tracking'
    )
    COMATLASSIANJIRACOREPROJECTTEMPLATESJIRACORESIMPLIFIEDPROCESSCONTROL = (
        'com.atlassian.jira-core-project-templates:jira-core-simplified-process-control'
    )
    COMATLASSIANJIRACOREPROJECTTEMPLATESJIRACORESIMPLIFIEDPROCUREMENT = (
        'com.atlassian.jira-core-project-templates:jira-core-simplified-procurement'
    )
    COMATLASSIANJIRACOREPROJECTTEMPLATESJIRACORESIMPLIFIEDPROJECTMANAGEMENT = 'com.atlassian.jira-core-project-templates:jira-core-simplified-project-management'
    COMATLASSIANJIRACOREPROJECTTEMPLATESJIRACORESIMPLIFIEDRECRUITMENT = (
        'com.atlassian.jira-core-project-templates:jira-core-simplified-recruitment'
    )
    COMATLASSIANJIRACOREPROJECTTEMPLATESJIRACORESIMPLIFIEDTASK = (
        'com.atlassian.jira-core-project-templates:jira-core-simplified-task-'
    )
    COMATLASSIANJCSCUSTOMERSERVICEMANAGEMENT = (
        'com.atlassian.jcs:customer-service-management'
    )


class CreateProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    assigneeType: SharePermissionAssigneeType = Field(default=None)
    avatarId: int = Field(default=None)
    categoryId: int = Field(default=None)
    description: str = Field(default=None)
    fieldConfigurationScheme: int = Field(default=None)
    fieldScheme: int = Field(default=None)
    issueSecurityScheme: int = Field(default=None)
    issueTypeScheme: int = Field(default=None)
    issueTypeScreenScheme: int = Field(default=None)
    key: str
    lead: str = Field(default=None)
    leadAccountId: str = Field(default=None, max_length=128)
    name: str
    notificationScheme: int = Field(default=None)
    permissionScheme: int = Field(default=None)
    projectTemplateKey: CreateProjectDetailsProjectTemplateKey = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    url: str = Field(default=None)
    workflowScheme: int = Field(default=None)


class ProjectIdentifiers(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int
    key: str
    self: str


class ProjectCreateResourceIdentifierType(_HtmlReprMixin, str, Enum):
    ID = 'id'
    REF = 'ref'


class ProjectCreateResourceIdentifier(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    anID: bool = Field(default=None)
    areference: bool = Field(default=None)
    entityId: str = Field(default=None)
    entityType: str = Field(default=None)
    id: str = Field(default=None)
    type_: ProjectCreateResourceIdentifierType = Field(default=None, alias='type')


class IssueTypeSchemePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultIssueTypeId: ProjectCreateResourceIdentifier = Field(default=None)
    description: str | None = Field(default=None)
    issueTypeIds: list[ProjectCreateResourceIdentifier] = Field(default=None)
    name: str = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)


class IssueTypeProjectCreatePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeHierarchy: Any | None = Field(default=None)
    issueTypeScheme: IssueTypeSchemePayload = Field(default=None)
    issueTypes: Any | None = Field(default=None)


class FieldLayoutSchemePayload(_HtmlReprMixin, BaseModel):
    """FieldLayoutSchemePayload is deprecated.

    .. deprecated::
        This model is deprecated."""

    model_config = {'extra': 'forbid'}
    defaultFieldLayout: ProjectCreateResourceIdentifier = Field(default=None)
    description: str = Field(default=None)
    explicitMappings: dict[str, ProjectCreateResourceIdentifier] = Field(default=None)
    name: str = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)


class IssueTypeScreenSchemePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultScreenScheme: ProjectCreateResourceIdentifier = Field(default=None)
    description: str = Field(default=None)
    explicitMappings: dict[str, ProjectCreateResourceIdentifier] = Field(default=None)
    name: str = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)


class WorkflowStatusLayoutPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    x: float = Field(default=None)
    y: float = Field(default=None)


class WorkflowStatusPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    layout: WorkflowStatusLayoutPayload = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)
    properties: dict[str, str] = Field(default=None)


class BoardFeaturePayloadState(_HtmlReprMixin, Enum):
    VALUE_True = True
    VALUE_False = False


class NonWorkingDay(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int = Field(default=None)
    iso8601Date: str = Field(default=None)


class WorkingDaysConfig(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    friday: bool = Field(default=None)
    id: int = Field(default=None)
    monday: bool = Field(default=None)
    nonWorkingDays: list[NonWorkingDay] = Field(default=None)
    saturday: bool = Field(default=None)
    sunday: bool = Field(default=None)
    thursday: bool = Field(default=None)
    timezoneId: str = Field(default=None)
    tuesday: bool = Field(default=None)
    wednesday: bool = Field(default=None)


class SecurityLevelMemberPayloadType(_HtmlReprMixin, str, Enum):
    GROUP = 'group'
    REPORTER = 'reporter'
    USERS = 'users'


class SecurityLevelMemberPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    parameter: str = Field(default=None)
    type_: SecurityLevelMemberPayloadType = Field(default=None, alias='type')


class FieldSchemePayloadOnConflict(_HtmlReprMixin, str, Enum):
    FAIL = 'FAIL'
    USE = 'USE'
    NEW = 'NEW'


class WorkflowSchemePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultWorkflow: ProjectCreateResourceIdentifier = Field(default=None)
    description: str = Field(default=None)
    explicitMappings: dict[str, ProjectCreateResourceIdentifier] = Field(default=None)
    name: str = Field(default=None)
    onConflict: FieldSchemePayloadOnConflict = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)


class RolePayloadType(_HtmlReprMixin, str, Enum):
    HIDDEN = 'HIDDEN'
    VIEWABLE = 'VIEWABLE'
    EDITABLE = 'EDITABLE'
    GUEST = 'GUEST'


class RolePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultActors: list[ProjectCreateResourceIdentifier] = Field(default=None)
    description: str = Field(default=None)
    name: str = Field(default=None)
    onConflict: FieldSchemePayloadOnConflict = Field(default='USE')
    pcri: ProjectCreateResourceIdentifier = Field(default=None)
    type_: RolePayloadType = Field(default=None, alias='type')


class FieldAssociationItemPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)
    qualifierId: ProjectCreateResourceIdentifier = Field(default=None)
    qualifierType: ProjectCreateResourceIdentifier = Field(default=None)
    rendererType: str = Field(default=None)
    required: bool = Field(default=None)


class FieldSchemePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    items: list[FieldAssociationItemPayload] = Field(default=None)
    name: str = Field(default=None)
    onConflict: FieldSchemePayloadOnConflict = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)


class FieldCapabilityPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customFieldDefinitions: Any | None = Field(default=None)
    fieldLayoutScheme: FieldLayoutSchemePayload | None = Field(default=None)
    fieldLayouts: Any | None = Field(default=None)
    fieldScheme: FieldSchemePayload | None = Field(default=None)
    issueLayouts: Any | None = Field(default=None)
    issueTypeScreenScheme: IssueTypeScreenSchemePayload | None = Field(default=None)
    screenScheme: Any | None = Field(default=None)
    screens: Any | None = Field(default=None)


class BoardFeaturePayloadFeatureKey(_HtmlReprMixin, str, Enum):
    ESTIMATION = 'ESTIMATION'
    SPRINTS = 'SPRINTS'


class BoardFeaturePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    featureKey: BoardFeaturePayloadFeatureKey = Field(default=None)
    state: BoardFeaturePayloadState = Field(default=None)


class CardLayoutFieldMode(_HtmlReprMixin, str, Enum):
    PLAN = 'PLAN'
    WORK = 'WORK'


class CustomTemplatesProjectDetailsAccessLevel(_HtmlReprMixin, str, Enum):
    OPEN = 'open'
    LIMITED = 'limited'
    PRIVATE = 'private'
    FREE = 'free'


class SwimlanesPayloadSwimlaneStrategy(_HtmlReprMixin, str, Enum):
    NONE = 'none'
    CUSTOM = 'custom'
    PARENTCHILD = 'parentChild'
    ASSIGNEE = 'assignee'
    ASSIGNEEUNASSIGNEDFIRST = 'assigneeUnassignedFirst'
    EPIC = 'epic'
    PROJECT = 'project'
    ISSUEPARENT = 'issueparent'
    ISSUECHILDREN = 'issuechildren'
    REQUESTTYPE = 'request_type'


class SwimlanePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    jqlQuery: str = Field(default=None)
    name: str = Field(default=None)


class BoardPayloadCardColorStrategy(_HtmlReprMixin, str, Enum):
    ISSUETYPE = 'ISSUE_TYPE'
    REQUESTTYPE = 'REQUEST_TYPE'
    ASSIGNEE = 'ASSIGNEE'
    PRIORITY = 'PRIORITY'
    NONE = 'NONE'
    CUSTOM = 'CUSTOM'


class CardLayoutField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str = Field(default=None)
    id: int = Field(default=None)
    mode: CardLayoutFieldMode = Field(default=None)
    position: int = Field(default=None)


class SwimlanesPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customSwimlanes: list[SwimlanePayload] = Field(default=None)
    defaultCustomSwimlaneName: str = Field(default=None)
    swimlaneStrategy: SwimlanesPayloadSwimlaneStrategy = Field(default=None)


class CardLayout(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    showDaysInColumn: BoardFeaturePayloadState = Field(default=False)


class QuickFilterPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    jqlQuery: str = Field(default=None)
    name: str = Field(default=None)


class BoardColumnPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    maximumIssueConstraint: int = Field(default=None)
    minimumIssueConstraint: int = Field(default=None)
    name: str = Field(default=None)
    statusIds: list[ProjectCreateResourceIdentifier] = Field(default=None)


class BoardPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    boardFilterJQL: str = Field(default=None)
    cardColorStrategy: BoardPayloadCardColorStrategy = Field(default=None)
    cardLayout: CardLayout = Field(default=None)
    cardLayouts: list[CardLayoutField] = Field(default=None)
    columns: list[BoardColumnPayload] = Field(default=None)
    features: list[BoardFeaturePayload] = Field(default=None)
    name: str = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)
    quickFilters: list[QuickFilterPayload] = Field(default=None)
    supportsSprint: bool = Field(default=True)
    swimlanes: SwimlanesPayload = Field(default=None)
    workingDaysConfig: WorkingDaysConfig = Field(default=None)


class FromLayoutPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fromPort: int = Field(default=None)
    status: ProjectCreateResourceIdentifier = Field(default=None)
    toPortOverride: int = Field(default=None)


class RulePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    parameters: dict[str, str] = Field(default=None)
    ruleKey: str = Field(default=None)


class ToLayoutPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    port: int = Field(default=None)
    status: ProjectCreateResourceIdentifier = Field(default=None)


class ConditionGroupPayloadOperation(_HtmlReprMixin, str, Enum):
    ANY = 'ANY'
    ALL = 'ALL'


class ConditionGroupPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    conditionGroup: list[ConditionGroupPayload] = Field(default=None)
    conditions: list[RulePayload] = Field(default=None)
    operation: ConditionGroupPayloadOperation = Field(default=None)


class TransitionPayloadType(_HtmlReprMixin, str, Enum):
    GLOBAL = 'global'
    INITIAL = 'initial'
    DIRECTED = 'directed'


class TransitionPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    actions: list[RulePayload] = Field(default=None)
    conditions: ConditionGroupPayload = Field(default=None)
    customIssueEventId: str = Field(default=None)
    description: str = Field(default=None)
    from_: list[FromLayoutPayload] = Field(default=None, alias='from')
    id: int = Field(default=None)
    name: str = Field(default=None)
    properties: dict[str, str] = Field(default=None)
    to: ToLayoutPayload = Field(default=None)
    transitionScreen: RulePayload = Field(default=None)
    triggers: list[RulePayload] = Field(default=None)
    type_: TransitionPayloadType = Field(default=None, alias='type')
    validators: list[RulePayload] = Field(default=None)


class WorkflowPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    loopedTransitionContainerLayout: WorkflowStatusLayoutPayload = Field(default=None)
    name: str = Field(default=None)
    onConflict: FieldSchemePayloadOnConflict = Field(default='NEW')
    pcri: ProjectCreateResourceIdentifier = Field(default=None)
    startPointLayout: WorkflowStatusLayoutPayload = Field(default=None)
    statuses: list[WorkflowStatusPayload] = Field(default=None)
    transitions: list[TransitionPayload] = Field(default=None)


class StatusPayloadStatusCategory(_HtmlReprMixin, str, Enum):
    TODO = 'TODO'
    INPROGRESS = 'IN_PROGRESS'
    DONE = 'DONE'


class StatusPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)
    onConflict: FieldSchemePayloadOnConflict = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)
    statusCategory: StatusPayloadStatusCategory = Field(default=None)


class WorkflowCapabilityPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statuses: list[StatusPayload] = Field(default=None)
    workflowScheme: WorkflowSchemePayload = Field(default=None)
    workflows: list[WorkflowPayload] = Field(default=None)


class NotificationSchemeNotificationDetailsPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    notificationType: str = Field(default=None)
    parameter: str = Field(default=None)


class NotificationSchemeEventIDPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)


class NotificationSchemeEventPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    event: NotificationSchemeEventIDPayload = Field(default=None)
    notifications: list[NotificationSchemeNotificationDetailsPayload] = Field(
        default=None
    )


class NotificationSchemePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)
    notificationSchemeEvents: list[NotificationSchemeEventPayload] = Field(default=None)
    onConflict: FieldSchemePayloadOnConflict = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)


class PermissionGrantDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    applicationAccess: list[str] = Field(default=None)
    groupCustomFields: list[ProjectCreateResourceIdentifier] = Field(default=None)
    groups: list[ProjectCreateResourceIdentifier] = Field(default=None)
    permissionKeys: list[str] = Field(default=None)
    projectRoles: list[ProjectCreateResourceIdentifier] = Field(default=None)
    specialGrants: list[str] = Field(default=None)
    userCustomFields: list[ProjectCreateResourceIdentifier] = Field(default=None)
    users: list[ProjectCreateResourceIdentifier] = Field(default=None)


class PermissionPayloadDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    addAddonRole: bool = Field(default=None)
    description: str = Field(default=None)
    grants: list[PermissionGrantDTO] = Field(default=None)
    name: str = Field(default=None)
    onConflict: FieldSchemePayloadOnConflict = Field(default='FAIL')
    pcri: ProjectCreateResourceIdentifier = Field(default=None)


class SecurityLevelPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    isDefault: BoardFeaturePayloadState = Field(default=None)
    name: str = Field(default=None)
    securityLevelMembers: list[SecurityLevelMemberPayload] = Field(default=None)


class RolesCapabilityPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    roleToProjectActors: dict[str, list[ProjectCreateResourceIdentifier]] = Field(
        default=None
    )
    roles: list[RolePayload] = Field(default=None)


class ScopePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    type_: UserPermissionType = Field(default=None, alias='type')


class BoardsPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    boards: list[BoardPayload] = Field(default=None)


class ProjectPayloadProjectTypeKey(_HtmlReprMixin, str, Enum):
    SOFTWARE = 'software'
    BUSINESS = 'business'
    SERVICEDESK = 'service_desk'
    PRODUCTDISCOVERY = 'product_discovery'


class ProjectPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldLayoutSchemeId: ProjectCreateResourceIdentifier = Field(default=None)
    issueSecuritySchemeId: ProjectCreateResourceIdentifier = Field(default=None)
    issueTypeSchemeId: ProjectCreateResourceIdentifier = Field(default=None)
    issueTypeScreenSchemeId: ProjectCreateResourceIdentifier = Field(default=None)
    notificationSchemeId: ProjectCreateResourceIdentifier = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)
    permissionSchemeId: ProjectCreateResourceIdentifier = Field(default=None)
    projectTypeKey: ProjectPayloadProjectTypeKey = Field(default=None)
    workflowSchemeId: ProjectCreateResourceIdentifier = Field(default=None)


class BoardFeaturesPayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    boardFeatures: dict[str, list[BoardFeaturePayload]] = Field(default=None)


class SecuritySchemePayload(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)
    pcri: ProjectCreateResourceIdentifier = Field(default=None)
    securityLevels: list[SecurityLevelPayload] = Field(default=None)


class CustomTemplateRequestDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    boardFeatures: BoardFeaturesPayload | None = Field(default=None)
    boards: BoardsPayload | None = Field(default=None)
    field: FieldCapabilityPayload | None = Field(default=None)
    issueType: IssueTypeProjectCreatePayload | None = Field(default=None)
    notification: NotificationSchemePayload | None = Field(default=None)
    permissionScheme: PermissionPayloadDTO | None = Field(default=None)
    project: ProjectPayload = Field(default=None)
    role: RolesCapabilityPayload | None = Field(default=None)
    scope: ScopePayload | None = Field(default=None)
    security: SecuritySchemePayload | None = Field(default=None)
    workflow: WorkflowCapabilityPayload | None = Field(default=None)


class CustomTemplatesProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accessLevel: CustomTemplatesProjectDetailsAccessLevel = Field(default=None)
    additionalProperties: dict[str, str] = Field(default=None)
    assigneeType: ProjectComponentAssigneeType = Field(default=None)
    avatarId: int = Field(default=None)
    categoryId: int = Field(default=None)
    description: str = Field(default=None)
    enableComponents: bool = Field(default=False)
    key: str = Field(default=None)
    language: str = Field(default=None)
    leadAccountId: str = Field(default=None)
    name: str = Field(default=None)
    url: str = Field(default=None)


class ProjectCustomTemplateCreateRequestDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    details: CustomTemplatesProjectDetails = Field(default=None)
    template: CustomTemplateRequestDTO = Field(default=None)


class CustomTemplateOptions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    enableScreenDelegatedAdminSupport: bool = Field(default=None)
    enableWorkflowDelegatedAdminSupport: bool = Field(default=None)


class EditTemplateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    templateDescription: str = Field(default=None, max_length=150)
    templateGenerationOptions: CustomTemplateOptions = Field(default=None)
    templateKey: str = Field(default=None)
    templateName: str = Field(default=None, max_length=50)


class ProjectArchetypeRealType(_HtmlReprMixin, str, Enum):
    BUSINESS = 'BUSINESS'
    SOFTWARE = 'SOFTWARE'
    PRODUCTDISCOVERY = 'PRODUCT_DISCOVERY'
    SERVICEDESK = 'SERVICE_DESK'
    CUSTOMERSERVICE = 'CUSTOMER_SERVICE'
    OPS = 'OPS'


class ProjectTemplateKey(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    key: str = Field(default=None)
    uuid: UUID = Field(default=None)


class ProjectTemplateModelType(_HtmlReprMixin, str, Enum):
    LIVE = 'LIVE'
    SNAPSHOT = 'SNAPSHOT'


class ProjectArchetype(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    realType: ProjectArchetypeRealType = Field(default=None)
    style: SharePermissionStyle = Field(default=None)
    type_: ProjectArchetypeRealType = Field(default=None, alias='type')


class ProjectTemplateModel(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    archetype: ProjectArchetype = Field(default=None)
    defaultBoardView: str = Field(default=None)
    description: str = Field(default=None)
    liveTemplateProjectIdReference: int = Field(default=None)
    name: str = Field(default=None)
    projectTemplateKey: ProjectTemplateKey = Field(default=None)
    snapshotTemplate: dict[str, dict[str, Any]] = Field(default=None)
    templateGenerationOptions: CustomTemplateOptions = Field(default=None)
    type_: ProjectTemplateModelType = Field(default=None, alias='type')


class SaveProjectTemplateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectId: int = Field(default=None)
    templateGenerationOptions: CustomTemplateOptions = Field(default=None)
    templateType: ProjectTemplateModelType = Field(default=None)


class SaveTemplateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    templateDescription: str = Field(default=None, max_length=150)
    templateFromProjectRequest: SaveProjectTemplateRequest = Field(default=None)
    templateName: str = Field(default=None, max_length=50)


class SaveTemplateResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectTemplateKey: ProjectTemplateKey = Field(default=None)


class AutoEnum20(_HtmlReprMixin, str, Enum):
    CATEGORY = 'category'
    CATEGORY_1 = '-category'
    CATEGORY_2 = '+category'
    KEY = 'key'
    KEY_1 = '-key'
    KEY_2 = '+key'
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'
    OWNER = 'owner'
    OWNER_1 = '-owner'
    OWNER_2 = '+owner'
    ISSUECOUNT = 'issueCount'
    ISSUECOUNT_1 = '-issueCount'
    ISSUECOUNT_2 = '+issueCount'
    LASTISSUEUPDATEDDATE = 'lastIssueUpdatedDate'
    LASTISSUEUPDATEDDATE_1 = '-lastIssueUpdatedDate'
    LASTISSUEUPDATEDDATE_2 = '+lastIssueUpdatedDate'
    ARCHIVEDDATE = 'archivedDate'
    ARCHIVEDDATE_1 = '+archivedDate'
    ARCHIVEDDATE_2 = '-archivedDate'
    DELETEDDATE = 'deletedDate'
    DELETEDDATE_1 = '+deletedDate'
    DELETEDDATE_2 = '-deletedDate'


class AutoEnum21(_HtmlReprMixin, str, Enum):
    VIEW = 'view'
    BROWSE = 'browse'
    EDIT = 'edit'
    CREATE = 'create'


class AutoEnum22(_HtmlReprMixin, str, Enum):
    LIVE = 'live'
    ARCHIVED = 'archived'
    DELETED = 'deleted'


class ProjectType(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    color: str = Field(default=None)
    descriptionI18nKey: str = Field(default=None)
    formattedKey: str = Field(default=None)
    icon: str = Field(default=None)
    key: str = Field(default=None)


class UpdateProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    assigneeType: SharePermissionAssigneeType = Field(default=None)
    avatarId: int = Field(default=None)
    categoryId: int = Field(default=None)
    description: str = Field(default=None)
    issueSecurityScheme: int = Field(default=None)
    key: str = Field(default=None)
    lead: str = Field(default=None)
    leadAccountId: str = Field(default=None, max_length=128)
    name: str = Field(default=None)
    notificationScheme: int = Field(default=None)
    permissionScheme: int = Field(default=None)
    releasedProjectKeys: list[str] = Field(default=None)
    url: str = Field(default=None)


class ProjectAvatars(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    custom: list[Avatar] = Field(default=None)
    system: list[Avatar] = Field(default=None)


class UpdateDefaultProjectClassificationBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str


class AutoEnum23(_HtmlReprMixin, str, Enum):
    DESCRIPTION = 'description'
    DESCRIPTION_1 = '-description'
    DESCRIPTION_2 = '+description'
    ISSUECOUNT = 'issueCount'
    ISSUECOUNT_1 = '-issueCount'
    ISSUECOUNT_2 = '+issueCount'
    LEAD = 'lead'
    LEAD_1 = '-lead'
    LEAD_2 = '+lead'
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'


class AutoEnum24(_HtmlReprMixin, str, Enum):
    JIRA = 'jira'
    COMPASS = 'compass'
    AUTO = 'auto'


class ComponentWithIssueCountUnnamedModel4(SimpleListWrapperGroupName):
    pass


class ComponentWithIssueCountUnnamedModel3(AvatarUrlsBean):
    pass


class ComponentWithIssueCountUser1(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: ComponentWithIssueCountUnnamedModel3 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: ComponentWithIssueCountUnnamedModel4 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class ComponentWithIssueCountUnnamedModel7(SimpleListWrapperGroupName):
    pass


class ComponentWithIssueCountUnnamedModel5(ComponentWithIssueCountUser1):
    pass


class ComponentWithIssueCountUnnamedModel1(SimpleListWrapperGroupName):
    pass


class ComponentWithIssueCountUnnamedModel6(AvatarUrlsBean):
    pass


class ComponentWithIssueCountUser2(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: ComponentWithIssueCountUnnamedModel6 = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: ComponentWithIssueCountUnnamedModel7 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class ComponentWithIssueCountUnnamedModel(AvatarUrlsBean):
    pass


class ComponentWithIssueCountUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: ComponentWithIssueCountUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: ComponentWithIssueCountUnnamedModel1 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class ComponentWithIssueCountUnnamedModel2(ComponentWithIssueCountUser):
    pass


class ComponentWithIssueCountUnnamedModel8(ComponentWithIssueCountUser2):
    pass


class ComponentWithIssueCount(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    assignee: ComponentWithIssueCountUnnamedModel2 = Field(default=None)
    assigneeType: ProjectComponentAssigneeType = Field(default=None)
    description: str = Field(default=None)
    id: str = Field(default=None)
    isAssigneeTypeValid: bool = Field(default=None)
    issueCount: int = Field(default=None)
    lead: ComponentWithIssueCountUnnamedModel5 = Field(default=None)
    name: str = Field(default=None)
    project: str = Field(default=None)
    projectId: int = Field(default=None)
    realAssignee: ComponentWithIssueCountUnnamedModel8 = Field(default=None)
    realAssigneeType: ProjectComponentAssigneeType = Field(default=None)
    self: str = Field(default=None)


class PageBeanComponentWithIssueCount(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ComponentWithIssueCount] = Field(default=None)


class ProjectFeatureState(_HtmlReprMixin, str, Enum):
    ENABLED = 'ENABLED'
    DISABLED = 'DISABLED'
    COMINGSOON = 'COMING_SOON'


class ProjectFeature(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    feature: str = Field(default=None)
    imageUri: str = Field(default=None)
    localisedDescription: str = Field(default=None)
    localisedName: str = Field(default=None)
    prerequisites: list[str] = Field(default=None)
    projectId: int = Field(default=None)
    state: ProjectFeatureState = Field(default=None)
    toggleLocked: bool = Field(default=None)


class ContainerForProjectFeatures(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    features: list[ProjectFeature] = Field(default=None)


class ActorsMap(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    group: list[str] = Field(default=None)
    groupId: list[str] = Field(default=None)
    user: list[str] = Field(default=None)


class ProjectRoleActorsUpdateBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    categorisedActors: dict[str, list[str]] = Field(default=None)
    id: int = Field(default=None)


class ProjectRoleDetailsUnnamedModel1(UpdatedProjectCategory):
    pass


class ProjectRoleDetailsUnnamedModel(AvatarUrlsBean):
    pass


class ProjectRoleDetailsProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: ProjectRoleDetailsUnnamedModel = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: ProjectRoleDetailsUnnamedModel1 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class ProjectRoleDetailsUnnamedModel2(ProjectRoleDetailsProjectDetails):
    pass


class ProjectRoleDetailsScope(_HtmlReprMixin, BaseModel):
    project: ProjectRoleDetailsUnnamedModel2 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class ProjectRoleDetailsUnnamedModel3(ProjectRoleDetailsScope):
    pass


class ProjectRoleDetailsType(_HtmlReprMixin, str, Enum):
    DEFAULT = 'DEFAULT'
    GUESTROLE = 'GUEST_ROLE'


class ProjectRoleDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    admin: bool = Field(default=None)
    default: bool = Field(default=None)
    description: str = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)
    roleConfigurable: bool = Field(default=None)
    scope: ProjectRoleDetailsUnnamedModel3 = Field(default=None)
    self: str = Field(default=None)
    translatedName: str = Field(default=None)
    type_: ProjectRoleDetailsType = Field(default=None, alias='type')


class IssueTypeWithStatus(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str
    name: str
    self: str
    statuses: list[StatusDetails]
    subtask: bool


class AutoEnum25(_HtmlReprMixin, str, Enum):
    DESCRIPTION = 'description'
    DESCRIPTION_1 = '-description'
    DESCRIPTION_2 = '+description'
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'
    RELEASEDATE = 'releaseDate'
    RELEASEDATE_1 = '-releaseDate'
    RELEASEDATE_2 = '+releaseDate'
    SEQUENCE = 'sequence'
    SEQUENCE_1 = '-sequence'
    SEQUENCE_2 = '+sequence'
    STARTDATE = 'startDate'
    STARTDATE_1 = '-startDate'
    STARTDATE_2 = '+startDate'


class PageBeanVersion(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Version] = Field(default=None)


class ProjectEmailAddress(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    emailAddress: str = Field(default=None)
    emailAddressStatus: list[str] = Field(default=None)


class IssueTypeInfo(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarId: int = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)


class ProjectIssueTypesHierarchyLevel(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    entityId: UUID = Field(default=None)
    issueTypes: list[IssueTypeInfo] = Field(default=None)
    level: int = Field(default=None)
    name: str = Field(default=None)


class ProjectIssueTypeHierarchy(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    hierarchy: list[ProjectIssueTypesHierarchyLevel] = Field(default=None)
    projectId: int = Field(default=None)


class IdBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int


class ProjectIssueSecurityLevels(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    levels: list[SecurityLevel]


class ProjectFieldBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    fieldId: str = Field(default=None)
    isRequired: bool = Field(default=None)
    projectId: int = Field(default=None)
    workTypeId: int = Field(default=None)


class PageBean2ProjectFieldBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ProjectFieldBean] = Field(default=None)


class ContentItemEntityType(_HtmlReprMixin, str, Enum):
    ISSUEFIELDVALUE = 'issuefieldvalue'
    ISSUECOMMENT = 'issue-comment'
    ISSUEWORKLOG = 'issue-worklog'


class ContentItem(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    entityId: str
    entityType: ContentItemEntityType
    id: str


class RedactionPosition(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    adfPointer: str = Field(default=None)
    expectedText: str
    from_: int = Field(alias='from')
    to: int


class SingleRedactionRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contentItem: ContentItem
    externalId: UUID
    reason: str
    redactionPosition: RedactionPosition


class BulkRedactionRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    redactions: list[SingleRedactionRequest] = Field(default=None)


class SingleRedactionResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    externalId: UUID
    successful: bool


class BulkRedactionResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    results: list[SingleRedactionResponse]


class RedactionJobStatusResponseJobStatus(_HtmlReprMixin, str, Enum):
    PENDING = 'PENDING'
    INPROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'


class RedactionJobStatusResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    bulkRedactionResponse: BulkRedactionResponse = Field(default=None)
    jobStatus: RedactionJobStatusResponseJobStatus = Field(default=None)


class Resolution(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class CreateResolutionDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None, max_length=255)
    name: str = Field(max_length=60)


class ResolutionId(_HtmlReprMixin, BaseModel):
    id: str


class SetDefaultResolutionRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str


class ReorderIssueResolutionsRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    after: str = Field(default=None)
    ids: list[str]
    position: str = Field(default=None)


class ResolutionJsonBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    default: bool = Field(default=None)
    description: str = Field(default=None)
    iconUrl: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)


class PageBeanResolutionJsonBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ResolutionJsonBean] = Field(default=None)


class UpdateResolutionDetails(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None, max_length=255)
    name: str = Field(max_length=60)


class CreateUpdateRoleRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)


class ActorInputBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    group: list[str] = Field(default=None)
    groupId: list[str] = Field(default=None)
    user: list[str] = Field(default=None)


class AutoEnum26(_HtmlReprMixin, str, Enum):
    GLOBAL = 'GLOBAL'
    TEMPLATE = 'TEMPLATE'
    PROJECT = 'PROJECT'


class ScreenUnnamedModel1(UpdatedProjectCategory):
    pass


class ScreenUnnamedModel(AvatarUrlsBean):
    pass


class ScreenProjectDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    avatarUrls: ScreenUnnamedModel = Field(default=None)
    id: str = Field(default=None)
    key: str = Field(default=None)
    name: str = Field(default=None)
    projectCategory: ScreenUnnamedModel1 = Field(default=None)
    projectTypeKey: IssueTypeDetailsProjectTypeKey = Field(default=None)
    self: str = Field(default=None)
    simplified: bool = Field(default=None)


class ScreenUnnamedModel2(ScreenProjectDetails):
    pass


class ScreenScope(_HtmlReprMixin, BaseModel):
    project: ScreenUnnamedModel2 = Field(default=None)
    type_: IssueTypeDetailsType = Field(default=None, alias='type')


class ScreenUnnamedModel3(ScreenScope):
    pass


class Screen(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: int = Field(default=None)
    name: str = Field(default=None)
    scope: ScreenUnnamedModel3 = Field(default=None)


class PageBeanScreen(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Screen] = Field(default=None)


class ScreenDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str


class UpdateScreenDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)


class ScreenableField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    name: str = Field(default=None)


class AddFieldBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fieldId: str


class MoveFieldBeanPosition(_HtmlReprMixin, str, Enum):
    EARLIER = 'Earlier'
    LATER = 'Later'
    FIRST = 'First'
    LAST = 'Last'


class MoveFieldBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    after: str = Field(default=None)
    position: MoveFieldBeanPosition = Field(default=None)


class ScreenTypes(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    create: int = Field(default=None)
    default: int
    edit: int = Field(default=None)
    view: int = Field(default=None)


class ScreenSchemeUnnamedModel1(ScreenTypes):
    pass


class ScreenSchemeUnnamedModel(PageBeanIssueTypeScreenScheme):
    pass


class ScreenScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: int = Field(default=None)
    issueTypeScreenSchemes: ScreenSchemeUnnamedModel = Field(default=None)
    name: str = Field(default=None)
    screens: ScreenSchemeUnnamedModel1 = Field(default=None)


class PageBeanScreenScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[ScreenScheme] = Field(default=None)


class ScreenSchemeDetailsUnnamedModel(ScreenTypes):
    pass


class ScreenSchemeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str
    screens: ScreenSchemeDetailsUnnamedModel


class ScreenSchemeId(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int


class UpdateScreenTypes(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    create: str = Field(default=None)
    default: str = Field(default=None)
    edit: str = Field(default=None)
    view: str = Field(default=None)


class UpdateScreenSchemeDetailsUnnamedModel(UpdateScreenTypes):
    pass


class UpdateScreenSchemeDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)
    screens: UpdateScreenSchemeDetailsUnnamedModel = Field(default=None)


class AutoEnum27(_HtmlReprMixin, str, Enum):
    STRICT = 'strict'
    WARN = 'warn'
    NONE = 'none'
    TRUE = 'true'
    FALSE = 'false'


class SearchResults(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: str = Field(default=None)
    issues: list[IssueBean] = Field(default=None)
    maxResults: int = Field(default=None)
    names: dict[str, str] = Field(default=None)
    schema: dict[str, JsonTypeBean] = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    warningMessages: list[str] = Field(default=None)


class SearchRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: list[str] = Field(default=None)
    fields: list[str] = Field(default=None)
    fieldsByKeys: bool = Field(default=None)
    jql: str = Field(default=None)
    maxResults: int = Field(default=50)
    properties: list[str] = Field(default=None)
    startAt: int = Field(default=None)
    validateQuery: AutoEnum27 = Field(default=None)


class JQLCountRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    jql: str = Field(default=None)


class JQLCountResultsBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    count: int = Field(default=None)


class SearchWarningLimitDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    actual: int = Field(default=None)
    arguments: str = Field(default=None)
    clause: str = Field(default=None)
    limit: int = Field(default=None)


class SearchWarningUnnamedModel(SearchWarningLimitDetails):
    pass


class SearchWarning(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    details: SearchWarningUnnamedModel = Field(default=None)
    message: str = Field(default=None)
    type_: str = Field(default=None, alias='type')


class SearchAndReconcileResults(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    issues: list[IssueBean] = Field(default=None)
    names: dict[str, str] = Field(default=None)
    nextPageToken: str = Field(default=None)
    schema: dict[str, JsonTypeBean] = Field(default=None)
    warnings: list[SearchWarning] = Field(default=None)


class SearchAndReconcileRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expand: str = Field(default=None)
    fields: list[str] = Field(default=None)
    fieldsByKeys: bool = Field(default=None)
    jql: str = Field(default=None)
    maxResults: int = Field(default=50)
    nextPageToken: str = Field(default=None)
    properties: list[str] = Field(default=None)
    reconcileIssues: list[int] = Field(default=None)


class HealthCheckResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)
    passed: bool = Field(default=None)


class ServerInformation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    baseUrl: str = Field(default=None)
    buildDate: datetime = Field(default=None)
    buildNumber: int = Field(default=None)
    deploymentType: str = Field(default=None)
    displayUrl: str = Field(default=None)
    displayUrlConfluence: str = Field(default=None)
    displayUrlServicedeskHelpCenter: str = Field(default=None)
    healthChecks: list[HealthCheckResult] = Field(default=None)
    scmInfo: str = Field(default=None)
    serverTime: datetime = Field(default=None)
    serverTimeZone: str = Field(default=None)
    serverTitle: str = Field(default=None)
    version: str = Field(default=None)
    versionNumbers: list[int] = Field(default=None)


class ProjectId(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str


class StatusScope(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    project: ProjectId | None = Field(default=None)
    type_: UserPermissionType = Field(alias='type')


class JiraStatus(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: StatusScope = Field(default=None)
    statusCategory: StatusPayloadStatusCategory = Field(default=None)


class StatusCreate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(max_length=255)
    statusCategory: StatusPayloadStatusCategory


class StatusCreateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    scope: StatusScope
    statuses: list[StatusCreate]


class StatusUpdate(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None)
    id: str
    name: str
    statusCategory: StatusPayloadStatusCategory


class StatusUpdateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statuses: list[StatusUpdate]


class PageOfStatuses(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[JiraStatus] = Field(default=None)


class StatusProjectIssueTypeUsage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)


class StatusProjectIssueTypeUsagePage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    nextPageToken: str = Field(default=None)
    values: list[StatusProjectIssueTypeUsage] = Field(default=None)


class StatusProjectIssueTypeUsageDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypes: StatusProjectIssueTypeUsagePage = Field(default=None)
    projectId: str = Field(default=None)
    statusId: str = Field(default=None)


class StatusProjectUsage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)


class StatusProjectUsagePage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    nextPageToken: str = Field(default=None)
    values: list[StatusProjectUsage] = Field(default=None)


class StatusProjectUsageDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projects: StatusProjectUsagePage = Field(default=None)
    statusId: str = Field(default=None)


class StatusWorkflowUsageWorkflow(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)


class StatusWorkflowUsagePage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    nextPageToken: str = Field(default=None)
    values: list[StatusWorkflowUsageWorkflow] = Field(default=None)


class StatusWorkflowUsageDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statusId: str = Field(default=None)
    workflows: StatusWorkflowUsagePage = Field(default=None)


class UiModificationContextDetailsViewType(_HtmlReprMixin, str, Enum):
    GIC = 'GIC'
    ISSUEVIEW = 'IssueView'
    ISSUETRANSITION = 'IssueTransition'
    JSMREQUESTCREATE = 'JSMRequestCreate'


class UiModificationContextDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    isAvailable: bool = Field(default=None)
    issueTypeId: str = Field(default=None)
    portalId: str = Field(default=None)
    projectId: str = Field(default=None)
    requestTypeId: str = Field(default=None)
    viewType: UiModificationContextDetailsViewType = Field(default=None)


class UiModificationDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contexts: list[UiModificationContextDetails] = Field(default=None)
    data: str = Field(default=None)
    description: str = Field(default=None)
    id: str
    name: str
    self: str


class PageBeanUiModificationDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[UiModificationDetails] = Field(default=None)


class CreateUiModificationDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contexts: list[UiModificationContextDetails] = Field(default=None)
    data: str = Field(default=None)
    description: str = Field(default=None)
    name: str


class UiModificationIdentifiers(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str
    self: str


class DetailedErrorCollection(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    details: dict[str, dict[str, Any]] = Field(default=None)
    errorMessages: list[str] = Field(default=None)
    errors: dict[str, str] = Field(default=None)


class UpdateUiModificationDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    contexts: list[UiModificationContextDetails] = Field(default=None)
    data: str = Field(default=None)
    description: str = Field(default=None)
    name: str = Field(default=None)


class AutoEnum28(_HtmlReprMixin, str, Enum):
    PROJECT = 'project'
    ISSUETYPE = 'issuetype'
    PRIORITY = 'priority'


class Avatars(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    custom: list[Avatar] = Field(default=None)
    system: list[Avatar] = Field(default=None)


class AutoEnum29(_HtmlReprMixin, str, Enum):
    XSMALL = 'xsmall'
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'
    XLARGE = 'xlarge'


class AutoEnum30(_HtmlReprMixin, str, Enum):
    PNG = 'png'
    SVG = 'svg'


class NewUserDetails(_HtmlReprMixin, BaseModel):
    applicationKeys: list[str] = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str
    key: str = Field(default=None)
    name: str = Field(default=None)
    password: str = Field(default=None)
    products: list[str]
    self: str = Field(default=None)


class PageBeanUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[User] = Field(default=None)


class UserMigrationBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None)
    key: str = Field(default=None)
    username: str = Field(default=None)


class UserColumnRequestBody(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    columns: list[str] = Field(default=None)


class UnrestrictedUserEmail(_HtmlReprMixin, BaseModel):
    accountId: str = Field(default=None)
    email: str = Field(default=None)


class UserKey(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    key: str = Field(default=None)


class PageBeanUserKey(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[UserKey] = Field(default=None)


class VersionMoveBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    after: str = Field(default=None)
    position: MoveFieldBeanPosition = Field(default=None)


class VersionUsageInCustomField(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customFieldId: int = Field(default=None)
    fieldName: str = Field(default=None)
    issueCountWithVersionInCustomField: int = Field(default=None)


class VersionIssueCounts(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customFieldUsage: list[VersionUsageInCustomField] = Field(default=None)
    issueCountWithCustomFieldsShowingVersion: int = Field(default=None)
    issuesAffectedCount: int = Field(default=None)
    issuesFixedCount: int = Field(default=None)
    self: str = Field(default=None)


class VersionRelatedWork(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    category: str
    issueId: int = Field(default=None)
    relatedWorkId: str = Field(default=None)
    title: str = Field(default=None)
    url: str = Field(default=None)


class CustomFieldReplacement(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customFieldId: int = Field(default=None)
    moveTo: int = Field(default=None)


class DeleteAndReplaceVersionBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    customFieldReplacementList: list[CustomFieldReplacement] = Field(default=None)
    moveAffectedIssuesTo: int = Field(default=None)
    moveFixIssuesTo: int = Field(default=None)


class VersionUnresolvedIssuesCount(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issuesCount: int = Field(default=None)
    issuesUnresolvedCount: int = Field(default=None)
    self: str = Field(default=None)


class WebhookEnum(_HtmlReprMixin, str, Enum):
    JIRAISSUECREATED = 'jira:issue_created'
    JIRAISSUEUPDATED = 'jira:issue_updated'
    JIRAISSUEDELETED = 'jira:issue_deleted'
    COMMENTCREATED = 'comment_created'
    COMMENTUPDATED = 'comment_updated'
    COMMENTDELETED = 'comment_deleted'
    ISSUEPROPERTYSET = 'issue_property_set'
    ISSUEPROPERTYDELETED = 'issue_property_deleted'
    SPRINTCREATED = 'sprint_created'
    SPRINTUPDATED = 'sprint_updated'
    SPRINTCLOSED = 'sprint_closed'
    SPRINTDELETED = 'sprint_deleted'
    SPRINTSTARTED = 'sprint_started'
    JIRAVERSIONRELEASED = 'jira:version_released'
    JIRAVERSIONUNRELEASED = 'jira:version_unreleased'
    JIRAVERSIONCREATED = 'jira:version_created'
    JIRAVERSIONMOVED = 'jira:version_moved'
    JIRAVERSIONUPDATED = 'jira:version_updated'
    JIRAVERSIONMERGED = 'jira:version_merged'
    JIRAVERSIONDELETED = 'jira:version_deleted'


class Webhook(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    events: list[WebhookEnum]
    expirationDate: int = Field(default=None)
    fieldIdsFilter: list[str] = Field(default=None)
    id: int
    issuePropertyKeysFilter: list[str] = Field(default=None)
    jqlFilter: str
    url: str


class PageBeanWebhook(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Webhook] = Field(default=None)


class WebhookDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    events: list[WebhookEnum]
    fieldIdsFilter: list[str] = Field(default=None)
    issuePropertyKeysFilter: list[str] = Field(default=None)
    jqlFilter: str


class WebhookRegistrationDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    url: str
    webhooks: list[WebhookDetails]


class RegisteredWebhook(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    createdWebhookId: int = Field(default=None)
    errors: list[str] = Field(default=None)


class ContainerForRegisteredWebhooks(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    webhookRegistrationResult: list[RegisteredWebhook] = Field(default=None)


class ContainerForWebhookIDs(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    webhookIds: list[int]


class FailedWebhook(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    body: str = Field(default=None)
    failureTime: int
    id: str
    url: str


class FailedWebhooks(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    maxResults: int
    next: str = Field(default=None)
    values: list[FailedWebhook]


class WebhooksExpirationDate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    expirationDate: int


class WorkflowHistoryReadRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    version: int = Field(default=None)
    workflowId: str = Field(default=None)


class WorkflowTrigger(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    parameters: dict[str, str]
    ruleKey: str


class WorkflowTransitionLinks(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fromPort: int | None = Field(default=None)
    fromStatusReference: str | None = Field(default=None)
    toPort: int | None = Field(default=None)


class WorkflowScope(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    project: ProjectId | None = Field(default=None)
    type_: UserPermissionType = Field(default=None, alias='type')


class WorkflowStatusLayout(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    x: float | None = Field(default=None)
    y: float | None = Field(default=None)


class WorkflowRuleConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str | None = Field(default=None)
    parameters: dict[str, str] = Field(default=None)
    ruleKey: str


class ConditionGroupConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    conditionGroups: list[ConditionGroupConfiguration] = Field(default=None)
    conditions: list[WorkflowRuleConfiguration] = Field(default=None)
    operation: ConditionGroupPayloadOperation = Field(default=None)


class WorkflowTransitionsType(_HtmlReprMixin, str, Enum):
    INITIAL = 'INITIAL'
    GLOBAL = 'GLOBAL'
    DIRECTED = 'DIRECTED'


class WorkflowTransitions(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    actions: list[WorkflowRuleConfiguration | None] = Field(default=None)
    conditions: ConditionGroupConfiguration | None = Field(default=None)
    customIssueEventId: str | None = Field(default=None)
    description: str = Field(default=None)
    id: str = Field(default=None)
    links: list[WorkflowTransitionLinks | None] = Field(default=None)
    name: str = Field(default=None)
    properties: dict[str, str] = Field(default=None)
    toStatusReference: str = Field(default=None)
    transitionScreen: WorkflowRuleConfiguration | None = Field(default=None)
    triggers: list[WorkflowTrigger] = Field(default=None)
    type_: WorkflowTransitionsType = Field(default=None, alias='type')
    validators: list[WorkflowRuleConfiguration] = Field(default=None)


class WorkflowLayout(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    x: float = Field(default=None)
    y: float = Field(default=None)


class ApprovalConfigurationConditionType(_HtmlReprMixin, str, Enum):
    NUMBER = 'number'
    PERCENT = 'percent'
    NUMBERPERPRINCIPAL = 'numberPerPrincipal'


class ApprovalConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    active: AutoEnum13
    conditionType: ApprovalConfigurationConditionType
    conditionValue: str
    exclude: Any | None = Field(default=None)
    fieldId: str
    prePopulatedFieldId: str | None = Field(default=None)
    transitionApproved: str
    transitionRejected: str


class WorkflowReferenceStatus(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    approvalConfiguration: ApprovalConfiguration | None = Field(default=None)
    deprecated: bool = Field(default=None)
    layout: WorkflowStatusLayout | None = Field(default=None)
    properties: dict[str, str] = Field(default=None)
    statusReference: str = Field(default=None)


class DocumentVersion(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    versionNumber: int = Field(default=None)


class WorkflowDocumentDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    created: str = Field(default=None)
    description: str = Field(default=None)
    id: str = Field(default=None)
    lastUpdateAuthorAAID: str = Field(default=None)
    loopedTransitionContainerLayout: WorkflowLayout | None = Field(default=None)
    name: str = Field(default=None)
    scope: WorkflowScope = Field(default=None)
    startPointLayout: WorkflowLayout | None = Field(default=None)
    statuses: list[WorkflowReferenceStatus] = Field(default=None)
    transitions: list[WorkflowTransitions] = Field(default=None)
    updated: str = Field(default=None)
    version: DocumentVersion = Field(default=None)


class WorkflowDocumentStatusDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: WorkflowScope = Field(default=None)
    statusCategory: str = Field(default=None)
    statusReference: str = Field(default=None)


class WorkflowHistoryReadResponseDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statuses: list[WorkflowDocumentStatusDTO] = Field(default=None)
    workflows: list[WorkflowDocumentDTO] = Field(default=None)


class WorkflowHistoryListRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    workflowId: str = Field(default=None)


class WorkflowHistoryItemDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isIntermediate: bool = Field(default=None)
    workflowId: str = Field(default=None)
    workflowVersion: int = Field(default=None)
    writtenAt: str = Field(default=None)


class WorkflowHistoryListResponseDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    entries: list[WorkflowHistoryItemDTO] = Field(default=None)


class AutoEnum31(_HtmlReprMixin, str, Enum):
    POSTFUNCTION = 'postfunction'
    CONDITION = 'condition'
    VALIDATOR = 'validator'


class WorkflowTransition(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: int
    name: str


class AppWorkflowTransitionRuleUnnamedModel(WorkflowTransition):
    pass


class WorkflowId(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    draft: bool = Field(default=None)
    name: str


class RuleConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    disabled: bool = Field(default=False)
    tag: str = Field(default=None, max_length=255)
    value: str


class AppWorkflowTransitionRule(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    configuration: RuleConfiguration
    id: str
    key: str
    transition: AppWorkflowTransitionRuleUnnamedModel = Field(default=None)


class WorkflowTransitionRules(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    conditions: list[AppWorkflowTransitionRule] = Field(default=None)
    postFunctions: list[AppWorkflowTransitionRule] = Field(default=None)
    validators: list[AppWorkflowTransitionRule] = Field(default=None)
    workflowId: WorkflowId


class PageBeanWorkflowTransitionRules(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[WorkflowTransitionRules] = Field(default=None)


class WorkflowTransitionRulesUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    workflows: list[WorkflowTransitionRules]


class WorkflowTransitionRulesUpdateErrorDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    ruleUpdateErrors: dict[str, list[str]]
    updateErrors: list[str]
    workflowId: WorkflowId


class WorkflowTransitionRulesUpdateErrors(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    updateResults: list[WorkflowTransitionRulesUpdateErrorDetails]


class WorkflowTransitionRulesDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    workflowId: WorkflowId
    workflowRuleIds: list[str]


class WorkflowsWithTransitionRulesDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    workflows: list[WorkflowTransitionRulesDetails]


class AutoEnum32(_HtmlReprMixin, str, Enum):
    NAME = 'name'
    NAME_1 = '-name'
    NAME_2 = '+name'
    CREATED = 'created'
    CREATED_1 = '-created'
    CREATED_2 = '+created'
    UPDATED = 'updated'
    UPDATED_1 = '+updated'
    UPDATED_2 = '-updated'


class PublishedWorkflowId(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    entityId: str = Field(default=None)
    name: str


class WorkflowOperations(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    canDelete: bool
    canEdit: bool


class TransitionScreenDetails(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str
    name: str = Field(default=None)


class WorkflowTransitionRule(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    configuration: dict[str, Any] = Field(default=None)
    type_: str = Field(alias='type')


class WorkflowRules(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    conditionsTree: WorkflowSimpleCondition | WorkflowCompoundCondition = Field(
        default=None
    )
    postFunctions: list[WorkflowTransitionRule] = Field(default=None)
    validators: list[WorkflowTransitionRule] = Field(default=None)


class Transition(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str
    from_: list[str] = Field(alias='from')
    id: str
    name: str
    properties: dict[str, dict[str, Any]] = Field(default=None)
    rules: WorkflowRules = Field(default=None)
    screen: TransitionScreenDetails = Field(default=None)
    to: str
    type_: TransitionPayloadType = Field(alias='type')


class WorkflowStatus(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str
    name: str
    properties: dict[str, dict[str, Any]] = Field(default=None)


class WorkflowSchemeIdName(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str
    name: str


class Workflow(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    created: datetime = Field(default=None)
    description: str
    hasDraftWorkflow: bool = Field(default=None)
    id: PublishedWorkflowId
    isDefault: bool = Field(default=None)
    operations: WorkflowOperations = Field(default=None)
    projects: list[ProjectDetails] = Field(default=None)
    schemes: list[WorkflowSchemeIdName] = Field(default=None)
    statuses: list[WorkflowStatus] = Field(default=None)
    transitions: list[Transition] = Field(default=None)
    updated: datetime = Field(default=None)


class PageBeanWorkflow(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[Workflow] = Field(default=None)


class WorkflowSimpleCondition(_HtmlReprMixin, BaseModel):
    configuration: dict[str, Any] = Field(default=None)
    nodeType: str
    type_: str = Field(alias='type')


class WorkflowCompoundConditionOperator(_HtmlReprMixin, str, Enum):
    AND = 'AND'
    OR = 'OR'


class WorkflowCompoundCondition(_HtmlReprMixin, BaseModel):
    conditions: list[WorkflowSimpleCondition | WorkflowCompoundCondition]
    nodeType: str
    operator: WorkflowCompoundConditionOperator


class AutoEnum33(_HtmlReprMixin, str, Enum):
    LIVE = 'live'
    DRAFT = 'draft'


class WorkflowTransitionProperty(_HtmlReprMixin, BaseModel):
    id: str = Field(default=None)
    key: str = Field(default=None)
    value: str


class WorkflowProjectIssueTypeUsage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)


class WorkflowProjectIssueTypeUsagePage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    nextPageToken: str = Field(default=None)
    values: list[WorkflowProjectIssueTypeUsage] = Field(default=None)


class WorkflowProjectIssueTypeUsageDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypes: WorkflowProjectIssueTypeUsagePage = Field(default=None)
    projectId: str = Field(default=None)
    workflowId: str = Field(default=None)


class ProjectUsage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)


class ProjectUsagePage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    nextPageToken: str = Field(default=None)
    values: list[ProjectUsage] = Field(default=None)


class WorkflowProjectUsageDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projects: ProjectUsagePage = Field(default=None)
    workflowId: str = Field(default=None)


class WorkflowSchemeUsage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)


class WorkflowSchemeUsagePage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    nextPageToken: str = Field(default=None)
    values: list[WorkflowSchemeUsage] = Field(default=None)


class WorkflowSchemeUsageDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    workflowId: str = Field(default=None)
    workflowSchemes: WorkflowSchemeUsagePage = Field(default=None)


class ProjectAndIssueTypePair(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str
    projectId: str


class WorkflowReadRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectAndIssueTypes: list[ProjectAndIssueTypePair] = Field(default=None)
    workflowIds: list[str] = Field(default=None)
    workflowNames: list[str] = Field(default=None)


class JiraWorkflowStatus(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    scope: WorkflowScope = Field(default=None)
    statusCategory: StatusPayloadStatusCategory = Field(default=None)
    statusReference: str = Field(default=None)


class JiraWorkflow(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    created: str | None = Field(default=None)
    description: str = Field(default=None)
    id: str = Field(default=None)
    isEditable: bool = Field(default=None)
    loopedTransitionContainerLayout: WorkflowLayout | None = Field(default=None)
    name: str = Field(default=None)
    scope: WorkflowScope = Field(default=None)
    startPointLayout: WorkflowLayout | None = Field(default=None)
    statuses: list[WorkflowReferenceStatus] = Field(default=None)
    taskId: str | None = Field(default=None)
    transitions: list[WorkflowTransitions] = Field(default=None)
    updated: str | None = Field(default=None)
    version: DocumentVersion = Field(default=None)


class WorkflowReadResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statuses: list[JiraWorkflowStatus] = Field(default=None)
    workflows: list[JiraWorkflow] = Field(default=None)


class AvailableWorkflowConnectRuleRuleType(_HtmlReprMixin, str, Enum):
    CONDITION = 'Condition'
    VALIDATOR = 'Validator'
    FUNCTION = 'Function'
    SCREEN = 'Screen'


class AvailableWorkflowSystemRule(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str
    incompatibleRuleKeys: list[str]
    isAvailableForInitialTransition: bool
    isVisible: bool
    name: str
    ruleKey: str
    ruleType: AvailableWorkflowConnectRuleRuleType


class WorkflowCapabilitiesEnum(_HtmlReprMixin, str, Enum):
    SOFTWARE = 'software'
    SERVICEDESK = 'service_desk'
    PRODUCTDISCOVERY = 'product_discovery'
    BUSINESS = 'business'
    UNKNOWN = 'unknown'


class AvailableWorkflowTriggerTypes(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    name: str = Field(default=None)
    type_: str = Field(default=None, alias='type')


class AvailableWorkflowTriggers(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    availableTypes: list[AvailableWorkflowTriggerTypes]
    ruleKey: str


class AvailableWorkflowForgeRule(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    ruleKey: str = Field(default=None)
    ruleType: AvailableWorkflowConnectRuleRuleType = Field(default=None)


class AvailableWorkflowConnectRule(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    addonKey: str = Field(default=None)
    createUrl: str = Field(default=None)
    description: str = Field(default=None)
    editUrl: str = Field(default=None)
    moduleKey: str = Field(default=None)
    name: str = Field(default=None)
    ruleKey: str = Field(default=None)
    ruleType: AvailableWorkflowConnectRuleRuleType = Field(default=None)
    viewUrl: str = Field(default=None)


class WorkflowCapabilities(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    connectRules: list[AvailableWorkflowConnectRule] = Field(default=None)
    editorScope: UserPermissionType = Field(default=None)
    forgeRules: list[AvailableWorkflowForgeRule] = Field(default=None)
    projectTypes: list[WorkflowCapabilitiesEnum] = Field(default=None)
    systemRules: list[AvailableWorkflowSystemRule] = Field(default=None)
    triggerRules: list[AvailableWorkflowTriggers] = Field(default=None)


class ConditionGroupUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    conditionGroups: list[ConditionGroupUpdate] = Field(default=None)
    conditions: list[WorkflowRuleConfiguration] = Field(default=None)
    operation: ConditionGroupPayloadOperation


class TransitionUpdateDTO(_HtmlReprMixin, BaseModel):
    actions: list[WorkflowRuleConfiguration] = Field(default=None)
    conditions: ConditionGroupUpdate | None = Field(default=None)
    customIssueEventId: str = Field(default=None)
    description: str = Field(default=None)
    id: str = Field(default=None)
    links: list[WorkflowTransitionLinks] = Field(default=None)
    name: str = Field(default=None)
    properties: dict[str, str] = Field(default=None)
    toStatusReference: str = Field(default=None)
    transitionScreen: WorkflowRuleConfiguration | None = Field(default=None)
    triggers: list[WorkflowTrigger] = Field(default=None)
    type_: WorkflowTransitionsType = Field(default=None, alias='type')
    validators: list[WorkflowRuleConfiguration] = Field(default=None)


class StatusLayoutUpdate(_HtmlReprMixin, BaseModel):
    approvalConfiguration: ApprovalConfiguration | None = Field(default=None)
    layout: WorkflowLayout | None = Field(default=None)
    properties: dict[str, str]
    statusReference: str


class WorkflowCreate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    loopedTransitionContainerLayout: WorkflowLayout | None = Field(default=None)
    name: str
    startPointLayout: WorkflowLayout | None = Field(default=None)
    statuses: list[StatusLayoutUpdate]
    transitions: list[TransitionUpdateDTO]


class WorkflowStatusUpdate(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str
    statusCategory: StatusPayloadStatusCategory
    statusReference: str


class WorkflowCreateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    scope: WorkflowScope = Field(default=None)
    statuses: list[WorkflowStatusUpdate] = Field(default=None, le=1000.0)
    workflows: list[WorkflowCreate] = Field(default=None, le=20.0)


class WorkflowCreateResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statuses: list[JiraWorkflowStatus] = Field(default=None)
    workflows: list[JiraWorkflow] = Field(default=None)


class ValidationOptionsForCreateEnum(_HtmlReprMixin, str, Enum):
    WARNING = 'WARNING'
    ERROR = 'ERROR'


class ValidationOptionsForCreate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    levels: list[ValidationOptionsForCreateEnum] = Field(default=None, max_length=2)


class WorkflowCreateValidateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    payload: WorkflowCreateRequest
    validationOptions: ValidationOptionsForCreate = Field(default=None)


class WorkflowElementReference(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    propertyKey: str = Field(default=None)
    ruleId: str = Field(default=None)
    statusMappingReference: ProjectAndIssueTypePair = Field(default=None)
    statusReference: str = Field(default=None)
    transitionId: str = Field(default=None)


class WorkflowValidationErrorType(_HtmlReprMixin, str, Enum):
    RULE = 'RULE'
    STATUS = 'STATUS'
    STATUSLAYOUT = 'STATUS_LAYOUT'
    STATUSPROPERTY = 'STATUS_PROPERTY'
    WORKFLOW = 'WORKFLOW'
    TRANSITION = 'TRANSITION'
    TRANSITIONPROPERTY = 'TRANSITION_PROPERTY'
    SCOPE = 'SCOPE'
    STATUSMAPPING = 'STATUS_MAPPING'
    TRIGGER = 'TRIGGER'


class WorkflowValidationError(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    additionalDetails: str = Field(default=None)
    code: str = Field(default=None)
    elementReference: WorkflowElementReference = Field(default=None)
    level: ValidationOptionsForCreateEnum = Field(default=None)
    message: str = Field(default=None)
    type_: WorkflowValidationErrorType = Field(default=None, alias='type')


class WorkflowValidationErrorList(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    errors: list[WorkflowValidationError] = Field(default=None)


class DefaultWorkflowEditorResponseValue(_HtmlReprMixin, str, Enum):
    NEW = 'NEW'
    LEGACY = 'LEGACY'


class DefaultWorkflowEditorResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    value: DefaultWorkflowEditorResponseValue = Field(default=None)


class WorkflowPreviewRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeIds: list[str] = Field(default=None, max_length=25)
    projectId: str
    workflowIds: list[str] = Field(default=None, max_length=25)
    workflowNames: list[str] = Field(default=None, max_length=25)


class ApprovalConfigurationPreview(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    active: str = Field(default=None)
    transitionApproved: str = Field(default=None)
    transitionRejected: str = Field(default=None)


class WorkflowPreviewLayout(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    x: float = Field(default=None)
    y: float = Field(default=None)


class WorkflowPreviewStatus(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    approvalConfiguration: ApprovalConfigurationPreview = Field(default=None)
    deprecated: bool = Field(default=None)
    layout: WorkflowPreviewLayout = Field(default=None)
    statusReference: str = Field(default=None)


class PreviewRuleConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    parameters: dict[str, str] = Field(default=None)
    ruleKey: str = Field(default=None)


class PreviewConditionGroupConfiguration(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    conditionGroups: list[PreviewConditionGroupConfiguration] = Field(default=None)
    conditions: list[PreviewRuleConfiguration] = Field(default=None)
    operation: ConditionGroupPayloadOperation = Field(default=None)


class TransitionLink(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    fromPort: int = Field(default=None)
    fromStatusReference: str = Field(default=None)
    toPort: int = Field(default=None)


class WorkflowProjectIdScope(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)


class WorkflowPreviewScope(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    project: WorkflowProjectIdScope | None = Field(default=None)
    type_: UserPermissionType = Field(default=None, alias='type')


class PreviewTrigger(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    ruleKey: str = Field(default=None)


class WorkflowDocumentVersionBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    id: str = Field(default=None)
    versionNumber: int = Field(default=None)


class TransitionPreview(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    actions: list[PreviewRuleConfiguration | None] = Field(default=None)
    conditions: PreviewConditionGroupConfiguration | None = Field(default=None)
    customIssueEventId: str = Field(default=None)
    description: str = Field(default=None)
    id: str = Field(default=None)
    links: list[TransitionLink] = Field(default=None)
    name: str = Field(default=None)
    toStatusReference: str = Field(default=None)
    transitionScreen: PreviewRuleConfiguration | None = Field(default=None)
    triggers: list[PreviewTrigger] = Field(default=None)
    type_: WorkflowTransitionsType = Field(default=None, alias='type')
    validators: list[PreviewRuleConfiguration] = Field(default=None)


class ProjectIssueTypeQueryContext(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypes: list[str] = Field(default=None)
    project: str = Field(default=None)


class WorkflowPreview(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    loopedTransitionContainerLayout: WorkflowPreviewLayout = Field(default=None)
    name: str = Field(default=None)
    queryContext: list[ProjectIssueTypeQueryContext] = Field(default=None)
    scope: WorkflowPreviewScope = Field(default=None)
    startPointLayout: WorkflowPreviewLayout = Field(default=None)
    statuses: list[WorkflowPreviewStatus] = Field(default=None)
    transitions: list[TransitionPreview] = Field(default=None)
    version: WorkflowDocumentVersionBean = Field(default=None)


class JiraWorkflowPreviewStatus(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)
    rawName: str = Field(default=None)
    scope: WorkflowPreviewScope = Field(default=None)
    statusCategory: StatusPayloadStatusCategory = Field(default=None)
    statusReference: str = Field(default=None)


class WorkflowPreviewResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statuses: list[JiraWorkflowPreviewStatus] = Field(default=None)
    workflows: list[WorkflowPreview] = Field(default=None)


class WorkflowSearchResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    statuses: list[JiraWorkflowStatus] = Field(default=None)
    total: int = Field(default=None)
    values: list[JiraWorkflow] = Field(default=None)


class StatusMigration(_HtmlReprMixin, BaseModel):
    newStatusReference: str
    oldStatusReference: str


class StatusMappingDTO(_HtmlReprMixin, BaseModel):
    issueTypeId: str
    projectId: str
    statusMigrations: list[StatusMigration]


class WorkflowUpdate(_HtmlReprMixin, BaseModel):
    defaultStatusMappings: list[StatusMigration] = Field(default=None)
    description: str = Field(default=None)
    id: str
    loopedTransitionContainerLayout: WorkflowLayout | None = Field(default=None)
    startPointLayout: WorkflowLayout | None = Field(default=None)
    statusMappings: list[StatusMappingDTO] = Field(default=None)
    statuses: list[StatusLayoutUpdate]
    transitions: list[TransitionUpdateDTO]
    version: DocumentVersion


class WorkflowUpdateRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statuses: list[WorkflowStatusUpdate] = Field(default=None, le=1000.0)
    workflows: list[WorkflowUpdate] = Field(default=None, le=20.0)


class WorkflowUpdateResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statuses: list[JiraWorkflowStatus] = Field(default=None)
    taskId: str | None = Field(default=None)
    workflows: list[JiraWorkflow] = Field(default=None)


class ValidationOptionsForUpdate(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    levels: list[ValidationOptionsForCreateEnum] = Field(default=None, max_length=2)


class WorkflowUpdateValidateRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    payload: WorkflowUpdateRequest
    validationOptions: ValidationOptionsForUpdate = Field(default=None)


class WorkflowSchemeUnnamedModel1(SimpleListWrapperGroupName):
    pass


class WorkflowSchemeUnnamedModel(AvatarUrlsBean):
    pass


class WorkflowSchemeUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: WorkflowSchemeUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: WorkflowSchemeUnnamedModel1 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class WorkflowSchemeUnnamedModel2(WorkflowSchemeUser):
    pass


class WorkflowScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultWorkflow: str = Field(default=None)
    description: str = Field(default=None)
    draft: bool = Field(default=None)
    id: int = Field(default=None)
    issueTypeMappings: dict[str, str] = Field(default=None)
    issueTypes: dict[str, IssueTypeDetails] = Field(default=None)
    lastModified: str = Field(default=None)
    lastModifiedUser: WorkflowSchemeUnnamedModel2 = Field(default=None)
    name: str = Field(default=None)
    originalDefaultWorkflow: str = Field(default=None)
    originalIssueTypeMappings: dict[str, str] = Field(default=None)
    self: str = Field(default=None)
    updateDraftIfNeeded: bool = Field(default=None)


class PageBeanWorkflowScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    isLast: bool = Field(default=None)
    maxResults: int = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    startAt: int = Field(default=None)
    total: int = Field(default=None)
    values: list[WorkflowScheme] = Field(default=None)


class WorkflowSchemeAssociationsUnnamedModel1(SimpleListWrapperGroupName):
    pass


class WorkflowSchemeAssociationsUnnamedModel(AvatarUrlsBean):
    pass


class WorkflowSchemeAssociationsUser(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    accountId: str = Field(default=None, max_length=128)
    accountType: AttachmentMetadataAccountType = Field(default=None)
    active: bool = Field(default=None)
    appType: str = Field(default=None)
    applicationRoles: UnnamedModel = Field(default=None)
    avatarUrls: WorkflowSchemeAssociationsUnnamedModel = Field(default=None)
    displayName: str = Field(default=None)
    emailAddress: str = Field(default=None)
    expand: str = Field(default=None)
    groups: WorkflowSchemeAssociationsUnnamedModel1 = Field(default=None)
    guest: bool = Field(default=None)
    key: str = Field(default=None)
    locale: str = Field(default=None)
    name: str = Field(default=None)
    self: str = Field(default=None)
    timeZone: str = Field(default=None)


class WorkflowSchemeAssociationsUnnamedModel2(WorkflowSchemeAssociationsUser):
    pass


class WorkflowSchemeAssociationsWorkflowScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultWorkflow: str = Field(default=None)
    description: str = Field(default=None)
    draft: bool = Field(default=None)
    id: int = Field(default=None)
    issueTypeMappings: dict[str, str] = Field(default=None)
    issueTypes: dict[str, IssueTypeDetails] = Field(default=None)
    lastModified: str = Field(default=None)
    lastModifiedUser: WorkflowSchemeAssociationsUnnamedModel2 = Field(default=None)
    name: str = Field(default=None)
    originalDefaultWorkflow: str = Field(default=None)
    originalIssueTypeMappings: dict[str, str] = Field(default=None)
    self: str = Field(default=None)
    updateDraftIfNeeded: bool = Field(default=None)


class WorkflowSchemeAssociationsUnnamedModel3(WorkflowSchemeAssociationsWorkflowScheme):
    pass


class WorkflowSchemeAssociations(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectIds: list[str]
    workflowScheme: WorkflowSchemeAssociationsUnnamedModel3


class ContainerOfWorkflowSchemeAssociations(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    values: list[WorkflowSchemeAssociations]


class WorkflowSchemeProjectAssociation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectId: str
    workflowSchemeId: str = Field(default=None)


class WorkflowAssociationStatusMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    newStatusId: str = Field(default=None)
    oldStatusId: str = Field(default=None)


class MappingsByIssueTypeOverride(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str = Field(default=None)
    statusMappings: list[WorkflowAssociationStatusMapping] = Field(default=None)


class WorkflowSchemeProjectSwitchBean(_HtmlReprMixin, BaseModel):
    mappingsByIssueTypeOverride: list[MappingsByIssueTypeOverride] = Field(default=None)
    projectId: str = Field(default=None)
    targetSchemeId: str = Field(default=None)


class WorkflowSchemeReadRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projectIds: Any | None = Field(default=None)
    workflowSchemeIds: Any | None = Field(default=None)


class WorkflowMetadataRestModel(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    description: str
    id: str
    name: str
    version: DocumentVersion


class WorkflowMetadataAndIssueTypeRestModel(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeIds: list[str]
    workflow: WorkflowMetadataRestModel


class WorkflowSchemeReadResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultWorkflow: WorkflowMetadataRestModel = Field(default=None)
    description: str | None = Field(default=None)
    id: str
    name: str
    scope: WorkflowScope
    taskId: str | None = Field(default=None)
    version: DocumentVersion
    workflowsForIssueTypes: list[WorkflowMetadataAndIssueTypeRestModel]


class WorkflowSchemeAssociation(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeIds: list[str]
    workflowId: str


class MappingsByWorkflow(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    newWorkflowId: str
    oldWorkflowId: str
    statusMappings: list[WorkflowAssociationStatusMapping]


class WorkflowSchemeUpdateRequest(_HtmlReprMixin, BaseModel):
    defaultWorkflowId: str = Field(default=None)
    description: str
    id: str
    name: str
    statusMappingsByIssueTypeOverride: list[MappingsByIssueTypeOverride] = Field(
        default=None
    )
    statusMappingsByWorkflows: list[MappingsByWorkflow] = Field(default=None)
    version: DocumentVersion
    workflowsForIssueTypes: list[WorkflowSchemeAssociation] = Field(default=None)


class WorkflowSchemeUpdateRequiredMappingsRequest(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultWorkflowId: str | None = Field(default=None)
    id: str
    workflowsForIssueTypes: list[WorkflowSchemeAssociation]


class StatusMetadata(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    category: StatusPayloadStatusCategory = Field(default=None)
    id: str = Field(default=None)
    name: str = Field(default=None)


class RequiredMappingByWorkflows(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    sourceWorkflowId: str = Field(default=None)
    statusIds: list[str] = Field(default=None)
    targetWorkflowId: str = Field(default=None)


class RequiredMappingByIssueType(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str = Field(default=None)
    statusIds: list[str] = Field(default=None)


class StatusesPerWorkflow(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    initialStatusId: str = Field(default=None)
    statuses: list[str] = Field(default=None)
    workflowId: str = Field(default=None)


class WorkflowSchemeUpdateRequiredMappingsResponse(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statusMappingsByIssueTypes: list[RequiredMappingByIssueType] = Field(default=None)
    statusMappingsByWorkflows: list[RequiredMappingByWorkflows] = Field(default=None)
    statuses: list[StatusMetadata] = Field(default=None)
    statusesPerWorkflow: list[StatusesPerWorkflow] = Field(default=None)


class DefaultWorkflow(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    updateDraftIfNeeded: bool = Field(default=None)
    workflow: str


class IssueTypeWorkflowMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueType: str = Field(default=None)
    updateDraftIfNeeded: bool = Field(default=None)
    workflow: str = Field(default=None)


class StatusMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueTypeId: str
    newStatusId: str
    statusId: str


class PublishDraftWorkflowScheme(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    statusMappings: list[StatusMapping] = Field(default=None)


class IssueTypesWorkflowMapping(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    defaultMapping: bool = Field(default=None)
    issueTypes: list[str] = Field(default=None)
    updateDraftIfNeeded: bool = Field(default=None)
    workflow: str = Field(default=None)


class WorkflowSchemeProjectUsageDTO(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    projects: ProjectUsagePage = Field(default=None)
    workflowSchemeId: str = Field(default=None)


class ChangedWorklog(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    properties: list[EntityProperty] = Field(default=None)
    updatedTime: int = Field(default=None)
    worklogId: int = Field(default=None)


class ChangedWorklogs(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    lastPage: bool = Field(default=None)
    nextPage: str = Field(default=None)
    self: str = Field(default=None)
    since: int = Field(default=None)
    until: int = Field(default=None)
    values: list[ChangedWorklog] = Field(default=None)


class OperationMessage(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    message: str
    statusCode: int


class ConnectModules(_HtmlReprMixin, BaseModel):
    modules: list[dict[str, Any]]


class ConnectCustomFieldValueType(_HtmlReprMixin, str, Enum):
    STRINGISSUEFIELD = 'StringIssueField'
    NUMBERISSUEFIELD = 'NumberIssueField'
    RICHTEXTISSUEFIELD = 'RichTextIssueField'
    SINGLESELECTISSUEFIELD = 'SingleSelectIssueField'
    MULTISELECTISSUEFIELD = 'MultiSelectIssueField'
    TEXTISSUEFIELD = 'TextIssueField'


class ConnectCustomFieldValue(_HtmlReprMixin, BaseModel):
    _type: ConnectCustomFieldValueType
    fieldID: int
    issueID: int
    number: float = Field(default=None)
    optionID: str = Field(default=None)
    richText: str = Field(default=None)
    string: str = Field(default=None)
    text: str = Field(default=None)


class ConnectCustomFieldValues(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    updateValueList: list[ConnectCustomFieldValue] = Field(default=None)


class AutoEnum34(_HtmlReprMixin, str, Enum):
    ISSUEPROPERTY = 'IssueProperty'
    COMMENTPROPERTY = 'CommentProperty'
    DASHBOARDITEMPROPERTY = 'DashboardItemProperty'
    ISSUETYPEPROPERTY = 'IssueTypeProperty'
    PROJECTPROPERTY = 'ProjectProperty'
    USERPROPERTY = 'UserProperty'
    WORKLOGPROPERTY = 'WorklogProperty'
    BOARDPROPERTY = 'BoardProperty'
    SPRINTPROPERTY = 'SprintProperty'


class EntityPropertyDetails(_HtmlReprMixin, BaseModel):
    entityId: float
    key: str
    value: str


class WorkflowRulesSearch(_HtmlReprMixin, BaseModel):
    expand: str = Field(default=None)
    ruleIds: list[UUID] = Field(min_length=1, max_length=10)
    workflowEntityId: UUID


class WorkflowRulesSearchDetails(_HtmlReprMixin, BaseModel):
    invalidRules: list[UUID] = Field(default=None)
    validRules: list[WorkflowTransitionRules] = Field(default=None)
    workflowEntityId: UUID = Field(default=None)


class TaskProgress(_HtmlReprMixin, BaseModel):
    description: str = Field(default=None)
    elapsedRuntime: int
    finished: datetime = Field(default=None)
    id: str
    lastUpdate: datetime
    message: str = Field(default=None)
    progress: int
    result: dict[str, Any] = Field(default=None)
    self: str
    started: datetime = Field(default=None)
    status: BulkOperationProgressStatus
    submitted: datetime = Field(default=None)
    submittedBy: int


class ServiceRegistryTier(_HtmlReprMixin, BaseModel):
    description: str | None = Field(default=None)
    id: UUID = Field(default=None)
    level: int = Field(default=None)
    name: str | None = Field(default=None)
    nameKey: str = Field(default=None)


class ServiceRegistry(_HtmlReprMixin, BaseModel):
    description: str | None = Field(default=None)
    id: UUID = Field(default=None)
    name: str = Field(default=None)
    organizationId: str = Field(default=None)
    revision: str = Field(default=None)
    serviceTier: ServiceRegistryTier = Field(default=None)


class getForgeAppPropertyKeysResponseUnnamedModel(_HtmlReprMixin, BaseModel):
    key: str = Field(default=None)
    self: str = Field(default=None)


class getForgeAppPropertyKeysResponseUnnamedModel1(_HtmlReprMixin, BaseModel):
    keys: list[getForgeAppPropertyKeysResponseUnnamedModel] = Field(default=None)


class getForgeAppPropertyResponseUnnamedModel(_HtmlReprMixin, BaseModel):
    key: str = Field(default=None)
    value: dict[str, Any] = Field(default=None)


class WorklogCompositeKey(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueId: int = Field(default=None)
    worklogId: int = Field(default=None)


class BulkWorklogKeyRequestBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    requests: list[WorklogCompositeKey] = Field(default=None)


class WorklogKeyResult(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    issueId: int = Field(default=None)
    worklogId: int = Field(default=None)


class BulkWorklogKeyResponseBean(_HtmlReprMixin, BaseModel):
    model_config = {'extra': 'forbid'}
    worklogs: list[WorklogKeyResult] = Field(default=None)


BulkIssueResults.model_rebuild()
CompoundClause.model_rebuild()
ConditionGroupConfiguration.model_rebuild()
ConditionGroupPayload.model_rebuild()
ConditionGroupUpdate.model_rebuild()
CustomTemplateRequestDTO.model_rebuild()
IssueBean.model_rebuild()
JqlQuery.model_rebuild()
LinkGroup.model_rebuild()
NotificationEvent.model_rebuild()
NotificationScheme.model_rebuild()
NotificationSchemeEvent.model_rebuild()
Operations.model_rebuild()
PageBeanNotificationScheme.model_rebuild()
PageBeanWorkflow.model_rebuild()
ParsedJqlQueries.model_rebuild()
ParsedJqlQuery.model_rebuild()
PreviewConditionGroupConfiguration.model_rebuild()
ProjectCustomTemplateCreateRequestDTO.model_rebuild()
Transition.model_rebuild()
TransitionPayload.model_rebuild()
TransitionPreview.model_rebuild()
TransitionUpdateDTO.model_rebuild()
Workflow.model_rebuild()
WorkflowCapabilityPayload.model_rebuild()
WorkflowCompoundCondition.model_rebuild()
WorkflowCreate.model_rebuild()
WorkflowCreateRequest.model_rebuild()
WorkflowDocumentDTO.model_rebuild()
WorkflowHistoryReadResponseDTO.model_rebuild()
WorkflowPayload.model_rebuild()
WorkflowPreview.model_rebuild()
WorkflowPreviewResponse.model_rebuild()
WorkflowRules.model_rebuild()
WorkflowTransitions.model_rebuild()
